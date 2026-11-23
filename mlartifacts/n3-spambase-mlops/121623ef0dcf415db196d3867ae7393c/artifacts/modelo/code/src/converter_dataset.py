"""Converte o arquivo original spambase.data (UCI, id 94) em CSV com os nomes oficiais.

Os nomes das 57 features são lidos do arquivo spambase.names (linhas no formato
"nome: continuous."). A coluna alvo recebe o nome definido em config (spam).
Executar uma única vez:  python -m src.converter_dataset
"""
from __future__ import annotations

import re

import pandas as pd

from src.utils import caminho, carregar_config


def ler_nomes_colunas(arquivo_names) -> list[str]:
    """Extrai os nomes das features do .names, ignorando comentários (linhas com |)."""
    nomes = []
    padrao = re.compile(r"^([^|\s][^:]*):\s*continuous\.\s*$")
    with open(arquivo_names, encoding="utf-8") as f:
        for linha in f:
            m = padrao.match(linha.strip())
            if m:
                nomes.append(m.group(1).strip())
    return nomes


def converter(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or carregar_config()
    d = cfg["dados"]
    features = ler_nomes_colunas(caminho(d["original_names"]))
    if len(features) != d["n_features_esperadas"]:
        raise ValueError(f"Esperadas {d['n_features_esperadas']} features, lidas {len(features)}")
    colunas = features + [d["alvo"]]
    df = pd.read_csv(caminho(d["original_data"]), header=None, names=colunas)
    if df.shape[1] != len(colunas):
        raise ValueError("Número de colunas do .data não confere com o .names")
    df[d["alvo"]] = df[d["alvo"]].astype(int)
    destino = caminho(d["bruto"])
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    return df


if __name__ == "__main__":
    df = converter()
    print(f"CSV gerado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(df["spam"].value_counts().rename({1: "spam", 0: "não spam"}))
