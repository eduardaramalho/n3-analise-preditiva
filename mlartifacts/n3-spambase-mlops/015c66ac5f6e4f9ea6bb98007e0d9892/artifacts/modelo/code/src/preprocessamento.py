"""Diagnóstico, tratamento e verificação de qualidade do treino (ETL).

Princípio contra vazamento: todo tratamento aqui é feito linha a linha, com regras fixas
(faixas válidas definidas na documentação do dataset), sem calcular estatísticas que
depois seriam usadas pelo modelo. Escalonamento e seleção de features ficam dentro do
Pipeline do scikit-learn, ajustados só nos dados de treino de cada fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Faixas válidas segundo spambase.DOCUMENTATION
PREFIXOS_PERCENTUAIS = ("word_freq_", "char_freq_")      # percentuais em [0, 100]
PREFIXO_MAIUSCULAS = "capital_run_length_"               # valores >= 1
COLUNAS_INTEIRAS = ("capital_run_length_longest", "capital_run_length_total")


def colunas_features(df: pd.DataFrame, cfg: dict) -> list[str]:
    """As 57 features: tudo que não é identificador, alvo ou coluna auxiliar."""
    excluir = {cfg["dados"]["coluna_id"], cfg["dados"]["alvo"], "lote_origem"}
    return [c for c in df.columns if c not in excluir]


def mascara_fora_da_faixa(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """True onde o valor viola a faixa documentada da coluna."""
    m = pd.DataFrame(False, index=df.index, columns=features)
    for c in features:
        x = pd.to_numeric(df[c], errors="coerce")
        if c.startswith(PREFIXOS_PERCENTUAIS):
            m[c] = (x < 0) | (x > 100)
        elif c.startswith(PREFIXO_MAIUSCULAS):
            m[c] = x < 1
            if c in COLUNAS_INTEIRAS:
                m[c] = m[c] | (x.notna() & (x % 1 != 0))
    return m


def grupos_conteudo(df: pd.DataFrame, features: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[features], index=False)


def diagnosticar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Diagnóstico por coluna e resumo geral do conjunto.

    Outliers são contados pela regra do IQR (acima de Q3 + 1,5 x IQR ou abaixo de
    Q1 - 1,5 x IQR) apenas para descrição. A coluna taxa_spam_outliers mostra a
    proporção de spam entre os e-mails marcados como outlier naquela coluna.
    """
    alvo, col_id = cfg["dados"]["alvo"], cfg["dados"]["coluna_id"]
    features = colunas_features(df, cfg)
    fora = mascara_fora_da_faixa(df, features)
    y = pd.to_numeric(df[alvo], errors="coerce")

    linhas = []
    for c in features:
        x = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = (x < lim_inf) | (x > lim_sup)
        linhas.append({
            "coluna": c,
            "tipo": str(df[c].dtype),
            "ausentes": int(df[c].isna().sum()),
            "nao_numericos": int((x.isna() & df[c].notna()).sum()),
            "fora_da_faixa": int(fora[c].sum()),
            "min": x.min(), "max": x.max(), "media": round(x.mean(), 4), "mediana": x.median(),
            "pct_zeros": round(100 * (x == 0).mean(), 2),
            "q1": round(q1, 4), "q3": round(q3, 4), "limite_sup_iqr": round(lim_sup, 4),
            "outliers_iqr": int(out.sum()),
            "pct_outliers": round(100 * out.mean(), 2),
            "taxa_spam_outliers": round(y[out].mean(), 4) if out.any() else np.nan,
        })
    tabela = pd.DataFrame(linhas)

    grupos = grupos_conteudo(df, features)
    rotulos_por_grupo = pd.DataFrame({"g": grupos, "y": y}).groupby("g")["y"].nunique()
    grupos_conflito = rotulos_por_grupo[rotulos_por_grupo > 1].index
    resumo = {
        "linhas": len(df),
        "colunas_features": len(features),
        "ausentes_total": int(df[features + [alvo]].isna().sum().sum()),
        "colunas_nao_numericas": int(sum(not pd.api.types.is_numeric_dtype(df[c]) for c in features)),
        "ids_duplicados": int(df[col_id].duplicated().sum()),
        "linhas_duplicadas_exatas": int(df[features + [alvo]].duplicated().sum()),
        "grupos_rotulo_conflitante": int(len(grupos_conflito)),
        "linhas_rotulo_conflitante": int(grupos.isin(grupos_conflito).sum()),
        "valores_fora_da_faixa": int(fora.values.sum()),
        "valores_alvo": sorted(y.dropna().unique().tolist()),
        "proporcao_spam": round(float(y.mean()), 4),
        "colunas_com_outliers_iqr": int((tabela["outliers_iqr"] > 0).sum()),
    }
    return tabela, resumo


def tratar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica os tratamentos em ordem e devolve (df_tratado, log_de_tratamentos)."""
    alvo, col_id = cfg["dados"]["alvo"], cfg["dados"]["coluna_id"]
    features = colunas_features(df, cfg)
    df = df.copy()
    log = []

    def registrar(etapa, mascara_remover, justificativa):
        nonlocal df
        n = int(mascara_remover.sum())
        df = df[~mascara_remover].reset_index(drop=True)
        log.append({"etapa": etapa, "linhas_removidas": n, "linhas_restantes": len(df),
                    "justificativa": justificativa})

    registrar("1. Identificador repetido", df[col_id].duplicated(),
              "O mesmo e-mail (id_email) não pode entrar duas vezes no treino.")

    for c in features + [alvo]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log.append({"etapa": "2. Conversão de tipos", "linhas_removidas": 0, "linhas_restantes": len(df),
                "justificativa": "Todas as features convertidas para numérico; texto inválido vira ausente e é tratado na etapa 4."})

    registrar("3. Alvo inválido", ~df[alvo].isin([0, 1]),
              "O alvo só pode ser 0 (não spam) ou 1 (spam). Sem rótulo confiável o registro não serve para treino.")
    df[alvo] = df[alvo].astype(int)

    registrar("4. Valores ausentes", df[features].isna().any(axis=1),
              "Remoção em vez de imputação: imputar exigiria estatísticas do conjunto (média, mediana) "
              "calculadas fora do Pipeline, o que geraria vazamento entre folds. A documentação indica que não há ausentes.")

    registrar("5. Fora da faixa documentada", mascara_fora_da_faixa(df, features).any(axis=1),
              "Frequências devem estar em [0, 100] e contagens de maiúsculas devem ser >= 1. Valor fora disso é erro de coleta.")

    grupos = grupos_conteudo(df, features)
    n_rotulos = df.groupby(grupos)[alvo].transform("nunique")
    registrar("6. Rótulo conflitante", (n_rotulos > 1).values,
              "E-mails com as 57 features idênticas mas rótulos diferentes são contraditórios: "
              "o modelo não tem como acertar os dois. Todas as cópias são removidas.")

    registrar("7. Duplicatas exatas", df[features + [alvo]].duplicated(keep="first").values,
              "Cópias idênticas dão peso extra a poucos e-mails e, na validação cruzada, a mesma linha "
              "pode cair no treino e na validação do fold, inflando a métrica. Mantida a primeira ocorrência.")

    log.append({"etapa": "8. Outliers", "linhas_removidas": 0, "linhas_restantes": len(df),
                "justificativa": "Mantidos. Em frequências de palavras e caracteres, valores altos são justamente o sinal "
                                 "de spam (ex.: muitos '$' ou '!'). Os modelos usados são robustos: Random Forest não é "
                                 "afetado pela escala e os modelos lineares recebem StandardScaler dentro do Pipeline."})
    return df, pd.DataFrame(log)


def validar_qualidade(df: pd.DataFrame, cfg: dict, colunas_esperadas: list[str]) -> pd.DataFrame:
    """Verificação automática de qualidade. Levanta ValueError se qualquer regra falhar."""
    alvo = cfg["dados"]["alvo"]
    features = colunas_features(df, cfg)
    regras = [
        ("Schema: colunas iguais às do dataset original, na mesma ordem", features == colunas_esperadas),
        (f"Quantidade de features = {cfg['dados']['n_features_esperadas']}", len(features) == cfg["dados"]["n_features_esperadas"]),
        ("Todas as features são numéricas", all(pd.api.types.is_numeric_dtype(df[c]) for c in features)),
        ("Alvo contém apenas 0 e 1", set(df[alvo].unique()) <= {0, 1}),
        ("Alvo contém as duas classes", {0, 1} <= set(df[alvo].unique())),
        ("Sem valores ausentes", not df[features + [alvo]].isna().any().any()),
        ("Sem valores fora da faixa documentada", not mascara_fora_da_faixa(df, features).values.any()),
        ("Sem identificadores repetidos", not df[cfg["dados"]["coluna_id"]].duplicated().any()),
        ("Treino não vazio", len(df) > 0),
    ]
    resultado = pd.DataFrame(regras, columns=["regra", "aprovado"])
    falhas = resultado.loc[~resultado["aprovado"], "regra"].tolist()
    if falhas:
        raise ValueError("Verificação de qualidade falhou: " + "; ".join(falhas))
    return resultado
