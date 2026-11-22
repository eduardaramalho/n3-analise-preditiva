"""Funções utilitárias compartilhadas: raiz do repositório, configuração e JSON."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_CONFIG = RAIZ / "config" / "config.yaml"


def carregar_config(caminho: str | Path = CAMINHO_CONFIG) -> dict[str, Any]:
    """Lê o config/config.yaml e devolve um dicionário."""
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def caminho(relativo: str | Path) -> Path:
    """Converte um caminho relativo à raiz do repositório em caminho absoluto."""
    return RAIZ / relativo


def fixar_seed(seed: int) -> None:
    """Fixa as sementes do Python e do NumPy para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)


def ler_json(relativo: str | Path, padrao: Any = None) -> Any:
    p = caminho(relativo)
    if not p.exists():
        return padrao
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def salvar_json(dados: Any, relativo: str | Path) -> Path:
    p = caminho(relativo)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=str)
    return p
