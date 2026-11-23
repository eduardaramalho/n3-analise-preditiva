"""Reinicia o estado do pipeline para testar os ciclos do zero.

Apaga: estado de ingestão, modelo vigente, histórico de modelos, artifacts, treinos
processados, figuras do relatório, evidências de execução e (opcionalmente) o MLflow.
NÃO apaga os dados brutos, o teste fixo nem os lotes.

Uso:
    python -m src.resetar_estado --confirmar
    python -m src.resetar_estado --confirmar --manter-mlflow
Em um notebook:
    from src.resetar_estado import resetar; resetar(confirmar=True)
"""
from __future__ import annotations

import argparse
import shutil

from src.utils import caminho, carregar_config


def _limpar_pasta(pasta) -> int:
    """Remove o conteúdo da pasta, preservando o .gitkeep."""
    if not pasta.exists():
        return 0
    n = 0
    for item in pasta.iterdir():
        if item.name == ".gitkeep":
            continue
        shutil.rmtree(item) if item.is_dir() else item.unlink()
        n += 1
    return n


def resetar(confirmar: bool = False, manter_mlflow: bool = False) -> list[str]:
    if not confirmar:
        raise RuntimeError("Passe confirmar=True (ou --confirmar) para apagar o estado.")
    cfg = carregar_config()
    c = cfg["caminhos"]
    removidos = []
    for arquivo in (c["estado_ingestao"], c["modelo_vigente"]):
        p = caminho(arquivo)
        if p.exists():
            p.unlink()
            removidos.append(arquivo)
    for pasta in (c["historico_modelos"], c["artifacts"], cfg["dados"]["pasta_processados"],
                  c["figuras"], f"{c['evidencias']}/execucoes"):
        if _limpar_pasta(caminho(pasta)):
            removidos.append(pasta + "/")
    if not manter_mlflow:
        m = cfg["mlflow"]
        db, arts = caminho(m["tracking_db"]), caminho(m["pasta_artefatos"])
        if db.exists():
            db.unlink()
            removidos.append(m["tracking_db"])
        if arts.exists():
            shutil.rmtree(arts)
            removidos.append(m["pasta_artefatos"] + "/")
    return removidos


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmar", action="store_true")
    ap.add_argument("--manter-mlflow", action="store_true")
    a = ap.parse_args()
    for r in resetar(a.confirmar, a.manter_mlflow) or ["(nada a remover)"]:
        print("removido:", r)
