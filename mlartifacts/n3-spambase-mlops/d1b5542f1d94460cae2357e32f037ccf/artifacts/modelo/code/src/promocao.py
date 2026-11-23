"""Eleição do vencedor, regra de promoção, manifesto do modelo vigente e histórico de modelos.

Regra de promoção (config.yaml, seção promocao):
- Sem vigente (primeiro ciclo): promove se F1 do desafiante >= f1_minimo.
- Com vigente: promove se F1 do desafiante >= f1_minimo E
  F1 do desafiante - F1 do vigente (reavaliado nos dados atuais) >= limiar_melhoria.
- Caso contrário, o vigente é mantido.
"""
from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone

import joblib
import pandas as pd

from src.features import SeletorFixo
from src.modelagem import construir_pipeline
from src.utils import caminho, ler_json, salvar_json

CASAS = 6  # arredondamento nas comparações, evita 0.004999999 por erro de ponto flutuante


def eleger_vencedor(resumos: dict[str, dict], metrica: str, direcao: str) -> str:
    """Escolhe o melhor modelo pela média da métrica principal na validação cruzada.

    Empate (mesma média com 6 casas): vence o de menor desvio padrão entre os folds;
    persistindo, o primeiro da lista (modelo A).
    """
    sinal = 1 if direcao == "maximizar" else -1
    ordem = sorted(
        resumos,
        key=lambda n: (-sinal * round(resumos[n][f"{metrica}_media"], CASAS), resumos[n][f"{metrica}_desvio"]),
    )
    return ordem[0]


def decidir_promocao(f1_desafiante: float, f1_vigente: float | None, cfg: dict) -> dict:
    p = cfg["promocao"]
    minimo, limiar = p["f1_minimo"], p["limiar_melhoria"]
    f1_d = round(f1_desafiante, CASAS)
    passa_minimo = f1_d >= minimo
    if f1_vigente is None:
        promover = passa_minimo
        ganho = None
        motivo = (f"Primeiro ciclo sem modelo vigente. F1 {f1_d:.4f} >= mínimo {minimo}: promovido."
                  if promover else
                  f"Primeiro ciclo sem modelo vigente. F1 {f1_d:.4f} < mínimo {minimo}: nenhum modelo promovido.")
    else:
        ganho = round(f1_d - round(f1_vigente, CASAS), CASAS)
        supera = ganho >= limiar
        promover = passa_minimo and supera
        if promover:
            motivo = f"F1 {f1_d:.4f} supera o vigente ({f1_vigente:.4f}) em {ganho:+.4f} (>= {limiar}) e atinge o mínimo {minimo}: promovido."
        elif not passa_minimo:
            motivo = f"F1 {f1_d:.4f} abaixo do mínimo {minimo}: vigente mantido."
        else:
            motivo = f"Ganho de F1 {ganho:+.4f} sobre o vigente ({f1_vigente:.4f}) é menor que o limiar {limiar}: vigente mantido."
    return {
        "promover": bool(promover),
        "decisao": "promovido" if promover else ("mantido_vigente" if f1_vigente is not None else "nenhum_promovido"),
        "motivo": motivo,
        "f1_desafiante": f1_d,
        "f1_vigente_reavaliado": None if f1_vigente is None else round(f1_vigente, CASAS),
        "ganho_f1": ganho,
        "passa_f1_minimo": bool(passa_minimo),
        "f1_minimo": minimo,
        "limiar_melhoria": limiar,
    }


# ---------------------------------------------------------------- manifesto do vigente

def carregar_vigente(cfg: dict) -> dict | None:
    """Devolve o manifesto, ou None se ainda não existe modelo vigente."""
    m = ler_json(cfg["caminhos"]["modelo_vigente"])
    if not m or not m.get("run_id"):
        return None
    return m


def reconstruir_pipeline_vigente(vigente: dict, seed: int):
    """Mesma configuração do vigente: algoritmo, hiperparâmetros e a lista de features DELE."""
    return construir_pipeline(vigente["algoritmo"], vigente["hiperparametros"], seed,
                              seletor=SeletorFixo(features=list(vigente["features"])))


def atualizar_manifesto(cfg: dict, decisao: dict, cycle_id: str, novo: dict | None) -> dict:
    """Grava models/modelo_vigente.json.

    Se promovido, os campos do modelo passam a ser os do novo vigente. Se não, os campos do
    modelo continuam os do vigente anterior. Em ambos os casos ultima_decisao é atualizada.
    """
    atual = ler_json(cfg["caminhos"]["modelo_vigente"], padrao={}) or {}
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ultima = {"cycle_id": cycle_id, "avaliado_em": agora, **decisao,
              "desafiante": None if novo is None else {k: novo[k] for k in ("run_id", "algoritmo", "nome_modelo")}}
    if decisao["promover"]:
        manifesto = {**novo, "cycle_id": cycle_id, "decisao": decisao["decisao"], "motivo": decisao["motivo"],
                     "promovido_em": agora}
    else:
        manifesto = {k: v for k, v in atual.items() if k not in ("ultima_decisao", "historico_decisoes")}
        if not manifesto:
            manifesto = {"run_id": None, "algoritmo": None, "cycle_id": None, "data_version": None,
                         "features": [], "metrica": None, "decisao": decisao["decisao"], "motivo": decisao["motivo"]}
    historico = list(atual.get("historico_decisoes", []))
    historico.append({"cycle_id": cycle_id, "decisao": decisao["decisao"], "motivo": decisao["motivo"],
                      "run_id_vigente_apos": manifesto.get("run_id")})
    manifesto["ultima_decisao"] = ultima
    manifesto["historico_decisoes"] = historico
    salvar_json(manifesto, cfg["caminhos"]["modelo_vigente"])
    return manifesto


# ---------------------------------------------------------------- histórico de modelos

def hash_modelo(pipeline) -> str:
    """Impressão digital do pipeline treinado. Mesmo treino + mesma configuração + seed = mesmo hash."""
    return hashlib.sha256(pickle.dumps(pipeline, protocol=4)).hexdigest()[:16]


def salvar_no_historico(pipeline, cfg: dict, algoritmo: str, hash_: str) -> tuple[str, bool]:
    """Salva em models/historico/<algoritmo>_<hash>.joblib (comprimido). Nunca sobrescreve.

    Devolve (caminho relativo, se o arquivo foi criado agora).
    """
    rel = f"{cfg['caminhos']['historico_modelos']}/{algoritmo}_{hash_}.joblib"
    destino = caminho(rel)
    if destino.exists():
        return rel, False
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destino, compress=3)
    return rel, True


def registrar_indice(cfg: dict, linhas: list[dict]) -> pd.DataFrame:
    """Acrescenta ao índice models/historico/indice_modelos.csv uma linha por modelo treinado no ciclo."""
    destino = caminho(f"{cfg['caminhos']['historico_modelos']}/indice_modelos.csv")
    novo = pd.DataFrame(linhas)
    if destino.exists():
        novo = pd.concat([pd.read_csv(destino), novo], ignore_index=True)
    novo.to_csv(destino, index=False)
    return novo


def carregar_modelo_vigente(cfg: dict, vigente: dict):
    """Carrega o pipeline do vigente a partir do arquivo no histórico (ou do MLflow, se faltar)."""
    arq = vigente.get("arquivo_modelo")
    if arq and caminho(arq).exists():
        return joblib.load(caminho(arq))
    import mlflow.sklearn
    return mlflow.sklearn.load_model(vigente["modelo_uri"])


def registrar_comparacao_ciclos(cfg: dict, linha: dict) -> pd.DataFrame:
    """Acrescenta o resumo do ciclo em reports/figuras/comparacao_ciclos.csv (um ciclo por linha)."""
    destino = caminho(cfg["caminhos"]["comparacao_ciclos"])
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela = pd.DataFrame([linha])
    if destino.exists():
        anterior = pd.read_csv(destino)
        anterior = anterior[anterior["cycle_id"] != linha["cycle_id"]]  # reexecução do mesmo ciclo
        tabela = pd.concat([anterior, tabela], ignore_index=True)
    tabela.to_csv(destino, index=False)
    return tabela
