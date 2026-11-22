"""Construção de modelos, pipelines e validação cruzada.

Regra contra vazamento: tudo o que aprende com os dados (seleção de features por votação,
StandardScaler e o próprio modelo) está dentro de um único Pipeline. Na validação cruzada
o Pipeline inteiro é reajustado em cada fold, usando apenas a parte de treino do fold.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ALGORITMOS = {
    "RandomForestClassifier": RandomForestClassifier,
    "LinearSVC": LinearSVC,
    "LogisticRegression": LogisticRegression,
}
# Modelos baseados em distância/coeficientes precisam de padronização; árvores não.
PRECISA_ESCALONAR = {"RandomForestClassifier": False, "LinearSVC": True, "LogisticRegression": True}


def construir_estimador(algoritmo: str, hiperparametros: dict | None, seed: int):
    """Instancia o algoritmo com os hiperparâmetros do config e a seed (se aceitar)."""
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"Algoritmo não suportado: {algoritmo}")
    classe = ALGORITMOS[algoritmo]
    params = dict(hiperparametros or {})
    if "random_state" in inspect.signature(classe).parameters:
        params["random_state"] = seed
    return classe(**params)


def construir_pipeline(algoritmo: str, hiperparametros: dict | None, seed: int, seletor=None) -> Pipeline:
    """Pipeline = [seletor de features] -> [StandardScaler, se o modelo precisar] -> modelo."""
    etapas = []
    if seletor is not None:
        etapas.append(("seletor", seletor))
    if PRECISA_ESCALONAR[algoritmo]:
        etapas.append(("escalonador", StandardScaler()))
    etapas.append(("modelo", construir_estimador(algoritmo, hiperparametros, seed)))
    return Pipeline(etapas)


def criar_validacao(cfg: dict) -> StratifiedKFold:
    v = cfg["validacao"]
    if v["estrategia"] != "StratifiedKFold":
        raise ValueError("Somente StratifiedKFold é suportado")
    return StratifiedKFold(n_splits=v["n_splits"], shuffle=v["shuffle"], random_state=cfg["projeto"]["seed"])


def _pontuacao_positiva(modelo, X):
    """Probabilidade (ou escore) da classe spam, usada no ROC AUC."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def calcular_metricas(y_true, y_pred, y_score) -> dict:
    """Métricas para a classe positiva (spam = 1)."""
    return {
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def validar_cruzado(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: dict) -> tuple[pd.DataFrame, list]:
    """Validação cruzada estratificada. Devolve métricas por fold e os pipelines de cada fold.

    Os folds são sempre os mesmos (seed fixa), então todos os modelos comparados
    são avaliados exatamente nas mesmas divisões.
    """
    linhas, ajustados = [], []
    for i, (idx_tr, idx_va) in enumerate(criar_validacao(cfg).split(X, y), start=1):
        p = clone(pipeline)
        p.fit(X.iloc[idx_tr], y.iloc[idx_tr])
        X_va, y_va = X.iloc[idx_va], y.iloc[idx_va]
        m = calcular_metricas(y_va, p.predict(X_va), _pontuacao_positiva(p, X_va))
        m["fold"] = i
        linhas.append(m)
        ajustados.append(p)
    return pd.DataFrame(linhas).set_index("fold"), ajustados


def resumir_folds(por_fold: pd.DataFrame) -> dict:
    """Média e desvio padrão de cada métrica entre os folds."""
    resumo = {}
    for m in por_fold.columns:
        resumo[f"{m}_media"] = float(np.mean(por_fold[m]))
        resumo[f"{m}_desvio"] = float(np.std(por_fold[m], ddof=1))
    return resumo
