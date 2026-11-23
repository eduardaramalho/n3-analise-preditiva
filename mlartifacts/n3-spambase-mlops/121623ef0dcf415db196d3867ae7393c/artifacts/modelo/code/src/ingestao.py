"""Ingestão incremental de lotes e identificação do ciclo.

Fluxo de um ciclo:
1. planejar_ciclo(): lê models/estado_ingestao.json e escolhe o próximo lote ainda não ingerido.
2. montar_treino(): junta os lotes do ciclo (os já ingeridos mais o novo), sem duplicar registros.
3. registrar_ciclo(): grava artifacts/ciclo_atual.json com cycle_id e data_version.
4. confirmar_ingestao(): chamado só no fim do pipeline (notebook 04). Se algum notebook falhar
   antes disso, o lote não é marcado como ingerido e o próximo ciclo tenta de novo.

O teste fixo (data/raw/teste_fixo.csv) nunca é lido aqui.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pandas as pd

from src.utils import caminho, ler_json, salvar_json

ESTADO_INICIAL = {"lotes_ingeridos": [], "historico": []}


def listar_lotes(cfg: dict) -> list[str]:
    """Nomes dos lotes disponíveis (sem extensão), em ordem de chegada."""
    pasta = caminho(cfg["dados"]["pasta_lotes"])
    return sorted(p.stem for p in pasta.glob("lote_*.csv"))


def carregar_estado(cfg: dict) -> dict:
    return ler_json(cfg["caminhos"]["estado_ingestao"], padrao=dict(ESTADO_INICIAL))


def gerar_cycle_id() -> str:
    """Identificador do ciclo: data e hora UTC, ex.: 20261117T230512Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def planejar_ciclo(cfg: dict) -> dict:
    """Decide quais lotes compõem o treino deste ciclo."""
    estado = carregar_estado(cfg)
    disponiveis = listar_lotes(cfg)
    ingeridos = [l for l in estado["lotes_ingeridos"] if l in disponiveis]
    pendentes = [l for l in disponiveis if l not in ingeridos]
    lote_novo = pendentes[0] if pendentes else None
    lotes_ciclo = ingeridos + ([lote_novo] if lote_novo else [])
    if not lotes_ciclo:
        raise RuntimeError("Nenhum lote disponível em data/raw/lotes. Rode src/preparar_lotes.py.")
    return {
        "cycle_id": gerar_cycle_id(),
        "lotes_ja_ingeridos": ingeridos,
        "lote_novo": lote_novo,
        "lotes_ciclo": lotes_ciclo,
        "tem_dados_novos": lote_novo is not None,
    }


def montar_treino(cfg: dict, lotes: list[str]) -> tuple[pd.DataFrame, dict]:
    """Concatena os lotes e remove registros repetidos pelo identificador (id_email).

    Um mesmo id_email nunca pode entrar duas vezes no treino, mesmo que um lote seja
    reenviado. Isso é diferente de e-mails com conteúdo idêntico, tratados no ETL.
    """
    pasta = caminho(cfg["dados"]["pasta_lotes"])
    col_id = cfg["dados"]["coluna_id"]
    partes = []
    for nome in lotes:
        parte = pd.read_csv(pasta / f"{nome}.csv")
        parte["lote_origem"] = nome
        partes.append(parte)
    df = pd.concat(partes, ignore_index=True)
    n_antes = len(df)
    df = df.drop_duplicates(subset=col_id, keep="first").reset_index(drop=True)
    info = {
        "linhas_por_lote": {n: int((df["lote_origem"] == n).sum()) for n in lotes},
        "linhas_concatenadas": n_antes,
        "ids_repetidos_removidos": n_antes - len(df),
        "linhas_apos_ingestao": len(df),
    }
    return df, info


def hash_treino(df: pd.DataFrame, col_id: str) -> str:
    """Hash SHA-256 (12 primeiros caracteres) do conteúdo do treino, ordenado por id."""
    conteudo = df.sort_values(col_id).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(conteudo).hexdigest()[:12]


def gerar_data_version(lotes: list[str], hash_: str) -> str:
    """Ex.: lotes-01+02_h3f2a9c1b7d4e. Identifica de forma única os dados do treino."""
    numeros = "+".join(l.replace("lote_", "") for l in lotes)
    return f"lotes-{numeros}_h{hash_}"


def registrar_ciclo(cfg: dict, ciclo: dict) -> None:
    salvar_json(ciclo, cfg["caminhos"]["ciclo_atual"])


def carregar_ciclo(cfg: dict) -> dict:
    ciclo = ler_json(cfg["caminhos"]["ciclo_atual"])
    if ciclo is None:
        raise RuntimeError("artifacts/ciclo_atual.json não existe. Execute o notebook 01 antes.")
    return ciclo


def atualizar_ciclo(cfg: dict, **campos) -> dict:
    """Acrescenta informações ao ciclo atual (ex.: status de cada notebook)."""
    ciclo = carregar_ciclo(cfg)
    ciclo.update(campos)
    registrar_ciclo(cfg, ciclo)
    return ciclo


def confirmar_ingestao(cfg: dict, ciclo: dict) -> dict:
    """Marca os lotes do ciclo como ingeridos. Só deve ser chamado no fim do pipeline."""
    estado = carregar_estado(cfg)
    estado["lotes_ingeridos"] = list(ciclo["lotes_ciclo"])
    # Reexecutar o notebook 04 no mesmo ciclo não duplica o registro
    estado["historico"] = [h for h in estado["historico"] if h["cycle_id"] != ciclo["cycle_id"]]
    estado["historico"].append({
        "cycle_id": ciclo["cycle_id"],
        "lote_novo": ciclo["lote_novo"],
        "data_version": ciclo["data_version"],
        "confirmado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    salvar_json(estado, cfg["caminhos"]["estado_ingestao"])
    return estado
