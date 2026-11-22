"""Separa o teste fixo e os lotes de chegada de dados. Executar UMA única vez.

    python -m src.preparar_lotes

Decisões:
- Cada e-mail recebe um identificador estável (id_email = posição no arquivo original),
  usado pela ingestão para nunca incorporar o mesmo registro duas vezes.
- O Spambase tem e-mails com features idênticas (391 linhas duplicadas). Se uma cópia
  ficasse no teste e outra no treino, o teste deixaria de ser independente. Por isso a
  divisão é feita por GRUPO de conteúdo: as linhas idênticas formam um grupo, a divisão
  estratificada (train_test_split com stratify) é feita sobre os grupos únicos e depois cada
  grupo leva todas as suas cópias. Assim cópias idênticas sempre caem no mesmo lado e a
  proporção de spam é preservada.
- 20% vira data/raw/teste_fixo.csv. Os 80% restantes viram lote_01 (75%) e lote_02 (25%).
"""
from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import caminho, carregar_config




def grupos_por_conteudo(X: pd.DataFrame) -> pd.Series:
    """Gera um identificador de grupo igual para linhas com features idênticas."""
    return pd.util.hash_pandas_object(X, index=False).astype(str)


def separar_fracao(df: pd.DataFrame, alvo: str, features: list[str], fracao: float, seed: int):
    """Separa uma fração estratificada do df, mantendo cópias idênticas no mesmo lado."""
    grupos = grupos_por_conteudo(df[features])
    tab = pd.DataFrame({"grupo": grupos.values, "y": df[alvo].values})
    # Um registro por grupo: rótulo da primeira ocorrência e tamanho do grupo
    unicos = tab.groupby("grupo", sort=False).agg(y=("y", "first"), n=("y", "size")).reset_index()
    # Estrato = rótulo x faixa de tamanho (1 cópia, 2 cópias, 3 ou mais). Sem a faixa de
    # tamanho, um único grupo grande de spam (há um com 69 cópias) desequilibra a proporção.
    faixa = pd.cut(unicos["n"], bins=[0, 1, 2, float("inf")], labels=["1", "2", "3+"]).astype(str)
    estrato = unicos["y"].astype(str) + "_" + faixa
    g_resto, g_parte = train_test_split(
        unicos["grupo"], test_size=fracao, stratify=estrato, random_state=seed
    )
    mask_parte = grupos.isin(set(g_parte)).values
    return df[~mask_parte].reset_index(drop=True), df[mask_parte].reset_index(drop=True)


def preparar(cfg: dict | None = None, sobrescrever: bool = False) -> dict:
    cfg = cfg or carregar_config()
    d, p = cfg["dados"], cfg["preparacao_lotes"]
    seed = cfg["projeto"]["seed"]
    alvo = d["alvo"]
    COLUNA_ID = d["coluna_id"]

    destino_teste = caminho(d["teste_fixo"])
    if destino_teste.exists() and not sobrescrever:
        raise FileExistsError("Lotes já preparados. Esta etapa roda uma única vez (use sobrescrever=True só se souber o que está fazendo).")

    df = pd.read_csv(caminho(d["bruto"]))
    df.insert(0, COLUNA_ID, range(len(df)))
    features = [c for c in df.columns if c not in (COLUNA_ID, alvo)]

    resto, teste = separar_fracao(df, alvo, features, p["proporcao_teste"], seed)

    props = p["proporcoes_lotes"]
    if len(props) != 2 or abs(sum(props) - 1) > 1e-9:
        raise ValueError("proporcoes_lotes deve ter dois valores que somam 1")
    lote_01, lote_02 = separar_fracao(resto, alvo, features, props[1], seed)

    pasta = caminho(d["pasta_lotes"])
    pasta.mkdir(parents=True, exist_ok=True)
    teste.sort_values(COLUNA_ID).to_csv(destino_teste, index=False)
    lote_01.sort_values(COLUNA_ID).to_csv(pasta / "lote_01.csv", index=False)
    lote_02.sort_values(COLUNA_ID).to_csv(pasta / "lote_02.csv", index=False)

    resumo = {}
    for nome, parte in [("teste_fixo", teste), ("lote_01", lote_01), ("lote_02", lote_02)]:
        resumo[nome] = {"linhas": len(parte), "proporcao_spam": round(parte[alvo].mean(), 4)}
    # Checagens de integridade
    ids = [set(x[COLUNA_ID]) for x in (teste, lote_01, lote_02)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(df)
    g = [set(grupos_por_conteudo(x[features])) for x in (teste, lote_01, lote_02)]
    assert not (g[0] & (g[1] | g[2])), "Há e-mail idêntico no teste e no treino"
    return resumo


if __name__ == "__main__":
    import json
    print(json.dumps(preparar(), indent=2, ensure_ascii=False))
