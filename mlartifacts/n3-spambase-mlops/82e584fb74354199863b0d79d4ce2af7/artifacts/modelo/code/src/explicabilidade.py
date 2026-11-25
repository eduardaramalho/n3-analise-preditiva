"""Explicabilidade do modelo vigente com SHAP.

O Pipeline do vigente é [seletor] -> [StandardScaler, se houver] -> modelo. O SHAP explica
o último passo (o classificador), recebendo os dados já transformados pelos passos anteriores.
- Random Forest: shap.TreeExplainer, valores em PROBABILIDADE de spam.
- Regressão Logística: shap.LinearExplainer, valores em LOG-ODDS (escala logit) de spam.
  O fundo (background) do LinearExplainer vem do TREINO, nunca do teste.
Os gráficos mostram os valores originais das features (antes do escalonamento),
para que a leitura faça sentido (ex.: char_freq_$ = 0,8 e não 2,3 desvios padrão).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def separar_pipeline(pipeline):
    """Devolve (transformação antes do modelo, modelo, seletor)."""
    return pipeline[:-1], pipeline.named_steps["modelo"], pipeline.named_steps["seletor"]


def explicar(pipeline, X: pd.DataFrame, X_fundo: pd.DataFrame, seed: int) -> tuple[shap.Explanation, str]:
    """Calcula os valores SHAP da classe spam para as linhas de X.

    Devolve (Explanation com os valores originais das features em .data, nome do explainer).
    """
    transformacao, modelo, seletor = separar_pipeline(pipeline)
    nomes = list(seletor.get_feature_names_out())
    X_modelo = pd.DataFrame(np.asarray(transformacao.transform(X), dtype=float), columns=nomes, index=X.index)
    X_original = seletor.transform(X)

    if hasattr(modelo, "estimators_"):          # Random Forest
        explainer = shap.TreeExplainer(modelo)
        valores = explainer(X_modelo, check_additivity=False)
        if valores.values.ndim == 3:             # (amostras, features, classes) -> classe spam
            valores = valores[:, :, 1]
        nome = "TreeExplainer (probabilidade de spam)"
    else:                                         # modelo linear
        fundo = pd.DataFrame(np.asarray(transformacao.transform(X_fundo), dtype=float), columns=nomes)
        explainer = shap.LinearExplainer(modelo, fundo)
        valores = explainer(X_modelo)
        nome = "LinearExplainer (log-odds de spam)"

    base = np.asarray(valores.base_values, dtype=float).reshape(-1)
    if base.size == 1:
        base = np.repeat(base, len(X))
    exp = shap.Explanation(values=np.asarray(valores.values, dtype=float), base_values=base,
                           data=X_original.to_numpy(dtype=float), feature_names=nomes)
    return exp, nome


def importancia_global(exp: shap.Explanation) -> pd.DataFrame:
    """Média do |SHAP| por feature: quanto, em média, cada feature move a previsão."""
    t = pd.DataFrame({"feature": exp.feature_names, "media_abs_shap": np.abs(exp.values).mean(axis=0),
                      "media_shap": exp.values.mean(axis=0)})
    return t.sort_values("media_abs_shap", ascending=False).reset_index(drop=True)


def escolher_exemplos(y_true: pd.Series, y_pred: np.ndarray, prob: np.ndarray, n: int = 3) -> list[tuple[str, int]]:
    """Escolhe e-mails para explicar: o spam mais "certo", o não spam mais "certo" e,
    se houver, o erro em que o modelo estava mais confiante (senão, o caso mais incerto).
    Devolve [(descrição, posição), ...].
    """
    y_true = np.asarray(y_true)
    acerto = y_true == y_pred
    escolhidos = []
    spam_ok = np.where(acerto & (y_true == 1))[0]
    if len(spam_ok):
        escolhidos.append(("spam classificado corretamente", int(spam_ok[np.argmax(prob[spam_ok])])))
    ham_ok = np.where(acerto & (y_true == 0))[0]
    if len(ham_ok):
        escolhidos.append(("não spam classificado corretamente", int(ham_ok[np.argmin(prob[ham_ok])])))
    erros = np.where(~acerto)[0]
    if len(erros):
        i = int(erros[np.argmax(np.abs(prob[erros] - 0.5))])
        tipo = "falso positivo (não spam marcado como spam)" if y_true[i] == 0 else "falso negativo (spam não detectado)"
        escolhidos.append((tipo, i))
    else:
        escolhidos.append(("caso mais incerto", int(np.argmin(np.abs(prob - 0.5)))))
    return escolhidos[:n]
