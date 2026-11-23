"""Padrão visual e funções para salvar tabelas (CSV) e gráficos (PNG) do relatório.

Todos os arquivos vão para reports/figuras/ com o cycle_id no início do nome,
por exemplo: 20261117T230512Z_01_diagnostico_antes.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from src.utils import caminho

# Paleta categórica validada para daltonismo (ordem fixa: azul, laranja, verde-água)
CORES = {"azul": "#2a78d6", "laranja": "#eb6834", "aqua": "#1baf7a", "cinza": "#8a8984"}
COR_CLASSE = {0: CORES["azul"], 1: CORES["laranja"]}   # 0 = não spam, 1 = spam
NOME_CLASSE = {0: "não spam", 1: "spam"}
TEXTO, TEXTO_SEC, FUNDO, GRADE = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def aplicar_estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "axes.edgecolor": GRADE, "axes.labelcolor": TEXTO_SEC, "axes.titlecolor": TEXTO,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SEC, "ytick.color": TEXTO_SEC, "font.size": 10,
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def _destino(cfg: dict, cycle_id: str, nome: str, extensao: str):
    pasta = caminho(cfg["caminhos"]["figuras"])
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta / f"{cycle_id}_{nome}.{extensao}"


def salvar_tabela(df: pd.DataFrame, cfg: dict, cycle_id: str, nome: str, index: bool = False):
    destino = _destino(cfg, cycle_id, nome, "csv")
    df.to_csv(destino, index=index, encoding="utf-8")
    return destino


def salvar_figura(fig, cfg: dict, cycle_id: str, nome: str):
    destino = _destino(cfg, cycle_id, nome, "png")
    fig.savefig(destino)
    return destino


def grafico_classes(contagens: dict[str, pd.Series], titulo: str):
    """Barras agrupadas da quantidade de e-mails por classe em cada etapa."""
    etapas = list(contagens)
    fig, ax = plt.subplots(figsize=(7, 4))
    largura = 0.38
    for i, classe in enumerate([0, 1]):
        valores = [int(contagens[e].get(classe, 0)) for e in etapas]
        xs = [j + (i - 0.5) * largura for j in range(len(etapas))]
        barras = ax.bar(xs, valores, width=largura - 0.02, color=COR_CLASSE[classe], label=NOME_CLASSE[classe])
        ax.bar_label(barras, padding=2, color=TEXTO_SEC, fontsize=9)
    ax.set_xticks(range(len(etapas)), etapas)
    ax.set_ylabel("e-mails")
    ax.set_title(titulo)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def grafico_spam_em_outliers(tabela: pd.DataFrame, taxa_geral: float, n: int = 15):
    """Taxa de spam entre os outliers (IQR) das colunas com mais outliers."""
    t = tabela.dropna(subset=["taxa_spam_outliers"]).nlargest(n, "outliers_iqr").sort_values("taxa_spam_outliers")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.5))
    ax.barh(t["coluna"], t["taxa_spam_outliers"], color=CORES["laranja"], height=0.7)
    ax.axvline(taxa_geral, color=TEXTO_SEC, linestyle="--", linewidth=1.2)
    ax.text(taxa_geral, len(t) - 0.4, f" taxa geral de spam ({taxa_geral:.0%})", color=TEXTO_SEC, fontsize=9, va="bottom")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("proporção de spam entre os outliers da coluna")
    ax.set_title(f"Outliers indicam spam: {len(t)} colunas com mais outliers (IQR)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


NOME_METODO = {"rf": "Random Forest", "svc": "LinearSVC", "lr": "Regressão Logística"}


def grafico_importancias(tabela_votos: pd.DataFrame, top_k: int):
    """Top-k de cada método, um painel por método (cada um com sua própria escala)."""
    fig, eixos = plt.subplots(1, 3, figsize=(15, 0.28 * top_k + 1.8))
    rotulos = {"rf": "importância (redução de impureza)", "svc": "|coeficiente| (dados padronizados)",
               "lr": "|coeficiente| (dados padronizados)"}
    for ax, s in zip(eixos, ["rf", "svc", "lr"]):
        t = tabela_votos.nsmallest(top_k, f"posicao_{s}").sort_values(f"posicao_{s}", ascending=False)
        cores = [CORES["azul"] if e else CORES["cinza"] for e in t["eleita"]]
        ax.barh(t["feature"], t[f"importancia_{s}"], color=cores, height=0.7)
        ax.set_title(f"{NOME_METODO[s]}: top {top_k}")
        ax.set_xlabel(rotulos[s])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=CORES["azul"], label="eleita na votação"),
                        Patch(color=CORES["cinza"], label="não eleita")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def grafico_votos(tabela_votos: pd.DataFrame, top_k: int):
    """Matriz feature x método: célula azul = está no top-k (1 voto). O número é a posição no ranking."""
    import numpy as np
    from matplotlib.colors import ListedColormap
    t = tabela_votos[tabela_votos["votos"] > 0].reset_index(drop=True)
    siglas = ["rf", "svc", "lr"]
    matriz = np.array([[1 if p <= top_k else 0 for p in t[f"posicao_{s}"]] for s in siglas]).T
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(t) + 1.6))
    ax.imshow(matriz, cmap=ListedColormap(["#f0efec", "#86b6ef"]), aspect="auto", vmin=0, vmax=1)
    for i in range(len(t)):
        for j, s in enumerate(siglas):
            ax.text(j, i, int(t.loc[i, f"posicao_{s}"]), ha="center", va="center", fontsize=8,
                    color=TEXTO if matriz[i, j] else TEXTO_SEC)
        ax.text(3.05, i, f"{int(t.loc[i, 'votos'])} voto(s)", va="center", fontsize=8, color=TEXTO,
                fontweight="bold" if t.loc[i, "eleita"] else "normal")
    ax.set_xticks(range(3), [NOME_METODO[s] for s in siglas])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(t)), t["feature"], fontsize=8)
    for lbl, eleita in zip(ax.get_yticklabels(), t["eleita"]):
        lbl.set_fontweight("bold" if eleita else "normal")
    n_eleitas = int(t["eleita"].sum())
    ax.axhline(n_eleitas - 0.5, color=CORES["laranja"], linewidth=1.5)
    ax.set_xlim(-0.5, 3.9)
    ax.grid(False)
    for lado in ("left", "bottom", "top", "right"):
        ax.spines[lado].set_visible(False)
    ax.set_title(f"Votação: posição no ranking de cada método (azul = top {top_k}).\n"
                 f"Acima da linha laranja: {n_eleitas} features eleitas (em negrito)", fontsize=10, pad=28)
    fig.tight_layout()
    return fig


def grafico_estabilidade(estabilidade: pd.DataFrame, n_folds: int):
    t = estabilidade.sort_values("folds_eleita")
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(t) + 1.5))
    cores = [CORES["azul"] if e else CORES["cinza"] for e in t["eleita_no_treino_completo"]]
    ax.barh(t["feature"], t["folds_eleita"], color=cores, height=0.7)
    ax.set_xlim(0, n_folds)
    ax.set_xticks(range(n_folds + 1))
    ax.set_xlabel(f"em quantos dos {n_folds} folds a feature foi eleita")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_title("Estabilidade da votação refeita no treino de cada fold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=CORES["azul"], label="eleita no treino completo"),
                       Patch(color=CORES["cinza"], label="não eleita no treino completo")],
              loc="lower right")
    fig.tight_layout()
    return fig


COR_MODELO = {"modelo_a": CORES["azul"], "modelo_b": CORES["laranja"], "vigente": CORES["aqua"]}
NOME_METRICA = {"f1": "F1 (spam)", "precision": "precisão", "recall": "recall", "accuracy": "acurácia", "roc_auc": "ROC AUC"}


def grafico_comparacao_modelos(tabela: pd.DataFrame, metricas: list[str], rotulos: dict[str, str]):
    """Barras agrupadas: média de cada métrica nos folds, com barra de erro = desvio padrão.

    tabela: índice = chave do modelo (modelo_a, modelo_b, vigente); colunas <metrica>_media e <metrica>_desvio.
    """
    import numpy as np
    modelos = list(tabela.index)
    n = len(modelos)
    largura = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, mod in enumerate(modelos):
        xs = np.arange(len(metricas)) + (i - (n - 1) / 2) * largura
        medias = [tabela.loc[mod, f"{m}_media"] for m in metricas]
        desvios = [tabela.loc[mod, f"{m}_desvio"] for m in metricas]
        barras = ax.bar(xs, medias, width=largura - 0.03, yerr=desvios, capsize=3, color=COR_MODELO[mod],
                        label=rotulos[mod], error_kw={"elinewidth": 1, "ecolor": TEXTO_SEC})
        ax.bar_label(barras, labels=[f"{v:.3f}" for v in medias], padding=4, fontsize=8, color=TEXTO_SEC)
    ax.set_xticks(range(len(metricas)), [NOME_METRICA[m] for m in metricas])
    mins = min(tabela[f"{m}_media"].min() for m in metricas)
    ax.set_ylim(max(0, mins - 0.08), 1.01)
    ax.set_ylabel("média nos 5 folds (barra de erro = desvio)")
    ax.set_title("Comparação dos modelos na validação cruzada (mesmos folds, mesmo treino)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n)
    fig.tight_layout()
    return fig


def grafico_f1_por_fold(por_fold: dict[str, pd.Series], rotulos: dict[str, str], f1_minimo: float):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for mod, serie in por_fold.items():
        ax.plot(serie.index, serie.values, marker="o", markersize=7, linewidth=2, color=COR_MODELO[mod],
                label=rotulos[mod])
    ax.axhline(f1_minimo, color=TEXTO_SEC, linestyle="--", linewidth=1.2)
    ax.text(ax.get_xlim()[0], f1_minimo, f" F1 mínimo para promoção ({f1_minimo})", va="bottom", fontsize=8, color=TEXTO_SEC)
    ax.set_xticks(list(next(iter(por_fold.values())).index))
    ax.set_xlabel("fold")
    ax.set_ylabel("F1 (spam)")
    ax.set_title("F1 em cada fold da validação cruzada")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    fig.tight_layout()
    return fig


def grafico_matriz_confusao(matriz, titulo: str):
    """Matriz de confusão com contagens e percentual da linha (classe real)."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#f0efec", "#86b6ef", "#256abf"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pct = matriz / matriz.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]}\n({pct[i, j]:.1%})", ha="center", va="center", fontsize=11,
                    color="white" if pct[i, j] > 0.6 else TEXTO)
    ax.set_xticks([0, 1], ["não spam", "spam"])
    ax.set_yticks([0, 1], ["não spam", "spam"])
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe real")
    ax.grid(False)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_title(titulo, fontsize=11)
    fig.tight_layout()
    return fig


def grafico_shap_global(importancia: pd.DataFrame, titulo: str, rotulo_x: str):
    t = importancia.sort_values("media_abs_shap")
    cores = [COR_CLASSE[1] if v > 0 else COR_CLASSE[0] for v in t["media_shap"]]
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(t) + 1.6))
    ax.barh(t["feature"], t["media_abs_shap"], color=cores, height=0.7)
    ax.set_xlabel(rotulo_x)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_title(titulo)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COR_CLASSE[1], label="em média empurra para spam"),
                       Patch(color=COR_CLASSE[0], label="em média empurra para não spam")], loc="lower right")
    fig.tight_layout()
    return fig


def grafico_evolucao_ciclos(historico: pd.DataFrame):
    """F1 do vigente reavaliado, dos candidatos e no teste fixo ao longo dos ciclos."""
    h = historico.reset_index(drop=True)
    x = range(len(h))
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(h) + 3), 4))
    series = [("f1_cv_modelo_a", "Modelo A (validação cruzada)", COR_MODELO["modelo_a"]),
              ("f1_cv_modelo_b", "Modelo B (validação cruzada)", COR_MODELO["modelo_b"]),
              ("f1_vigente_reavaliado", "Vigente reavaliado (validação cruzada)", "#6250d6"),
              ("teste_f1", "Vigente no teste fixo", COR_MODELO["vigente"])]
    for col, rotulo, cor in series:
        if col in h and h[col].notna().any():
            ax.plot(x, h[col], marker="o", markersize=7, linewidth=2, color=cor, label=rotulo)
    for i, d in enumerate(h["decisao"]):
        if d == "promovido":
            ax.axvline(i, color=GRADE, linewidth=6, zorder=0)
    ax.set_xticks(list(x), [f"{c[:8]}\n{c[9:15]}\n{l if isinstance(l, str) else 'sem lote'}"
                            for c, l in zip(h["cycle_id"], h["lote_novo"])], fontsize=8)
    ax.set_ylabel("F1 (spam)")
    ax.margins(y=0.15)
    ax.set_title("Evolução por ciclo (faixa cinza = ciclo com promoção)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.tight_layout()
    return fig
