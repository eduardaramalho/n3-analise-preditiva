"""Seleção de features por votação de três modelos.

Cada método gera um ranking completo das 57 features:
- RandomForestClassifier: feature_importances_ (redução média de impureza);
- LinearSVC e LogisticRegression: valor absoluto de coef_, com StandardScaler antes
  (sem padronização, o tamanho do coeficiente dependeria da escala da variável).

Regra de votação (parâmetros em config.yaml, seção votacao):
1. Cada método dá 1 voto às suas top_k features.
2. É eleita a feature com votos >= votos_minimos.
3. Desempate: média das posições nos 3 rankings (menor = melhor). Ele define a ordem
   das eleitas e quem entra quando é preciso completar o mínimo.
4. Se houver menos de minimo_features eleitas, completa-se com as não eleitas
   ordenadas por (votos, média das posições).

SeletorPorVotacao é um transformador do scikit-learn: ao ser colocado dentro de um
Pipeline, a votação é refeita só com os dados de treino de cada fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.modelagem import construir_pipeline, criar_validacao

SIGLAS = {"RandomForestClassifier": "rf", "LinearSVC": "svc", "LogisticRegression": "lr"}


def importancias(pipeline, nomes: list[str]) -> pd.Series:
    modelo = pipeline.named_steps["modelo"]
    if hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
    else:
        valores = np.abs(modelo.coef_).ravel()
    return pd.Series(valores, index=nomes, name="importancia")


def ranking(imp: pd.Series) -> pd.DataFrame:
    """Ranking completo: posição 1 = mais importante. Empates resolvidos pela ordem original."""
    ordem = imp.reset_index().rename(columns={"index": "feature"})
    ordem["ordem_original"] = range(len(ordem))
    ordem = ordem.sort_values(["importancia", "ordem_original"], ascending=[False, True])
    ordem["posicao"] = range(1, len(ordem) + 1)
    return ordem[["feature", "importancia", "posicao"]].reset_index(drop=True)


def apurar_votos(rankings: dict[str, pd.DataFrame], top_k: int, votos_minimos: int,
                 minimo_features: int) -> tuple[pd.DataFrame, list[str]]:
    """Monta a tabela feature x método x total e aplica a regra de eleição."""
    tabela = None
    for metodo, r in rankings.items():
        s = SIGLAS.get(metodo, metodo)
        parte = r.set_index("feature")[["importancia", "posicao"]].rename(
            columns={"importancia": f"importancia_{s}", "posicao": f"posicao_{s}"})
        parte[f"voto_{s}"] = (parte[f"posicao_{s}"] <= top_k).astype(int)
        tabela = parte if tabela is None else tabela.join(parte)
    siglas = [SIGLAS.get(m, m) for m in rankings]
    tabela["votos"] = tabela[[f"voto_{s}" for s in siglas]].sum(axis=1)
    tabela["media_posicoes"] = tabela[[f"posicao_{s}" for s in siglas]].mean(axis=1).round(2)
    tabela = tabela.sort_values(["votos", "media_posicoes"], ascending=[False, True])

    eleitas_por_voto = tabela.index[tabela["votos"] >= votos_minimos].tolist()
    if len(eleitas_por_voto) >= minimo_features:
        eleitas = eleitas_por_voto
        motivo = {f: "votos" for f in eleitas}
    else:
        eleitas = tabela.index[:minimo_features].tolist()
        motivo = {f: ("votos" if f in eleitas_por_voto else "completar_minimo") for f in eleitas}
    tabela["eleita"] = tabela.index.isin(eleitas)
    tabela["motivo"] = [motivo.get(f, "") for f in tabela.index]
    tabela["ordem_final"] = [eleitas.index(f) + 1 if f in eleitas else np.nan for f in tabela.index]
    tabela.index.name = "feature"
    return tabela.reset_index(), eleitas


class SeletorPorVotacao(BaseEstimator, TransformerMixin):
    """Transformador que elege features pela votação de três modelos.

    Parâmetros
    ----------
    modelos_votacao : dict  {nome_do_algoritmo: hiperparâmetros}, vindo do config.
    top_k, votos_minimos, minimo_features : regra de votação.
    seed : semente usada nos três modelos.
    """

    def __init__(self, modelos_votacao=None, top_k=20, votos_minimos=2, minimo_features=10, seed=42):
        self.modelos_votacao = modelos_votacao
        self.top_k = top_k
        self.votos_minimos = votos_minimos
        self.minimo_features = minimo_features
        self.seed = seed

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        nomes = list(X.columns)
        self.feature_names_in_ = np.array(nomes, dtype=object)
        self.rankings_ = {}
        for algoritmo, params in self.modelos_votacao.items():
            p = construir_pipeline(algoritmo, params, self.seed)
            p.fit(X, y)
            self.rankings_[algoritmo] = ranking(importancias(p, nomes))
        self.tabela_votos_, self.features_eleitas_ = apurar_votos(
            self.rankings_, self.top_k, self.votos_minimos, self.minimo_features)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X
        return X[self.features_eleitas_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.features_eleitas_, dtype=object)


def criar_seletor(cfg: dict) -> SeletorPorVotacao:
    v = cfg["votacao"]
    return SeletorPorVotacao(modelos_votacao=v["modelos"], top_k=v["top_k"], votos_minimos=v["votos_minimos"],
                             minimo_features=v["minimo_features"], seed=cfg["projeto"]["seed"])


class SeletorFixo(BaseEstimator, TransformerMixin):
    """Seleciona uma lista fixa de features (usado para reavaliar o modelo vigente)."""

    def __init__(self, features=None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.features)]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.features, dtype=object)


def estabilidade_votacao(seletores_por_fold: list[SeletorPorVotacao], eleitas_finais: list[str]) -> pd.DataFrame:
    """Em quantos folds cada feature foi eleita quando a votação rodou só no treino do fold."""
    n = len(seletores_por_fold)
    contagem = pd.Series(0, index=list(eleitas_finais), dtype=float)
    for s in seletores_por_fold:
        contagem = contagem.add(pd.Series(1, index=s.features_eleitas_), fill_value=0)
    t = contagem.astype(int).rename("folds_eleita").to_frame()
    t["pct_folds"] = (t["folds_eleita"] / n).round(3)
    t["eleita_no_treino_completo"] = t.index.isin(eleitas_finais)
    t.index.name = "feature"
    return t.sort_values(["folds_eleita", "eleita_no_treino_completo"], ascending=False).reset_index()


def votar_por_fold(cfg: dict, X: pd.DataFrame, y: pd.Series) -> list[SeletorPorVotacao]:
    """Refaz a votação no treino de cada fold da validação cruzada (mesmos folds do notebook 03)."""
    seletores = []
    for idx_tr, _ in criar_validacao(cfg).split(X, y):
        seletores.append(criar_seletor(cfg).fit(X.iloc[idx_tr], y.iloc[idx_tr]))
    return seletores
