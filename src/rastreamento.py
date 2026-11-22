"""Configuração do MLflow com backend SQLite portátil entre ambientes.

Problema: o MLflow grava no banco o caminho ABSOLUTO dos artefatos (ex.:
file:///home/runner/work/.../mlartifacts/...). Como o repositório fica em pastas
diferentes no Colab (/content/...), no runner do GitHub e no seu computador, esses
caminhos quebram ao trocar de ambiente.

Solução: antes de usar o MLflow, corrigir_caminhos() reescreve no mlflow.db todo
prefixo anterior a "mlartifacts/" para a raiz atual. O banco e os artefatos ficam
versionados no repositório e funcionam em qualquer ambiente.

Uso pela linha de comando (antes de abrir a interface):
    python -m src.rastreamento
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import logging
import re
import sqlite3
import warnings

from src.utils import caminho, carregar_config

TABELAS_CAMINHOS = [("experiments", "experiment_id", "artifact_location"), ("runs", "run_uuid", "artifact_uri")]


def corrigir_caminhos(cfg: dict) -> int:
    """Reescreve os caminhos de artefatos do mlflow.db para a raiz atual. Devolve quantos mudaram."""
    db = caminho(cfg["mlflow"]["tracking_db"])
    if not db.exists():
        return 0
    pasta = cfg["mlflow"]["pasta_artefatos"]
    destino = caminho(pasta).as_uri()
    padrao = re.compile(rf"^.*?/{re.escape(pasta)}(?=/|$)")
    alterados = 0
    with sqlite3.connect(db) as con:
        existentes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for tabela, chave, coluna in TABELAS_CAMINHOS:
            if tabela not in existentes:
                continue
            for id_, valor in con.execute(f"SELECT {chave}, {coluna} FROM {tabela}").fetchall():
                if not valor:
                    continue
                novo = padrao.sub(destino, valor, count=1)
                if novo != valor:
                    con.execute(f"UPDATE {tabela} SET {coluna} = ? WHERE {chave} = ?", (novo, id_))
                    alterados += 1
    return alterados


def configurar_mlflow(cfg: dict) -> str:
    """Aponta o MLflow para o sqlite do repositório e garante o experimento. Devolve o experiment_id."""
    import mlflow

    warnings.filterwarnings("ignore", module="mlflow")
    for nome_logger in ("alembic", "mlflow.store.db.utils"):
        logging.getLogger(nome_logger).setLevel(logging.WARNING)
    corrigir_caminhos(cfg)
    db = caminho(cfg["mlflow"]["tracking_db"])
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    nome = cfg["mlflow"]["experimento"]
    exp = mlflow.get_experiment_by_name(nome)
    if exp is None:
        local = f"{caminho(cfg['mlflow']['pasta_artefatos']).as_uri()}/{nome}"
        exp_id = mlflow.create_experiment(nome, artifact_location=local)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(nome)
    return exp_id


if __name__ == "__main__":
    n = corrigir_caminhos(carregar_config())
    print(f"Caminhos de artefatos ajustados para esta pasta: {n} registro(s) alterado(s).")


def registrar_modelo_sem_duplicar(pipeline, hash_modelo: str, X_exemplo, cfg: dict) -> dict:
    """Registra o pipeline completo no run ativo, sem guardar cópias idênticas.

    Se algum run anterior já guardou um modelo com o mesmo hash (mesmo treino, mesma
    configuração e mesma seed), este run apenas aponta para ele, pelas tags
    modelo_uri e modelo_reaproveitado_de. Caso contrário, salva o modelo com
    mlflow.sklearn.log_model (pipeline inteiro: seletor, escalonador e modelo).
    """
    import mlflow
    import mlflow.sklearn

    run_id = mlflow.active_run().info.run_id
    existentes = mlflow.search_runs(
        filter_string=f"tags.modelo_hash = '{hash_modelo}' and tags.modelo_salvo = 'sim'",
        output_format="list", max_results=1)
    if existentes:
        origem = existentes[0].info.run_id
        uri = f"runs:/{origem}/modelo"
        mlflow.set_tags({"modelo_hash": hash_modelo, "modelo_salvo": "reaproveitado",
                         "modelo_reaproveitado_de": origem, "modelo_uri": uri})
        return {"modelo_uri": uri, "reaproveitado_de": origem}
    mlflow.sklearn.log_model(
        pipeline, artifact_path="modelo", input_example=X_exemplo,
        pip_requirements=str(caminho("requirements.txt")), code_paths=[str(caminho("src"))])
    uri = f"runs:/{run_id}/modelo"
    mlflow.set_tags({"modelo_hash": hash_modelo, "modelo_salvo": "sim", "modelo_uri": uri})
    return {"modelo_uri": uri, "reaproveitado_de": None}
