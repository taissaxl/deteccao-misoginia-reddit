import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figuras", exist_ok=True)
AZUL, LARANJA = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#8a8985", "axes.linewidth": 0.6})

# métodos mostrados (as três variantes do BERTimbau tiveram resultados idênticos: fica uma linha só)
METODOS = {"BERTimbau (normalizado)": "BERTimbau e\ncascata",
           "Léxico sozinho": "Léxico sozinho",
           "LLM (Gemini 3.5 Flash Lite)": "LLM (Gemini)"}


def ler_ic(texto):
    """'[0.125; 0.800]' -> (0.125, 0.8)"""
    a, b = texto.strip("[]").split(";")
    return float(a), float(b)


def virgula(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


# --- figura 1: métricas com intervalo de confiança ---------------------------
conjuntos = [("resultados/avaliacao_metricas.csv", "Conjunto de avaliação (221 postagens)", AZUL, "o", AZUL, 0.13),
             ("resultados/avaliacao_metricas_300.csv", "Piloto + avaliação (278 postagens)", LARANJA, "s", "white", -0.13)]
tabelas = [pd.read_csv(arq).set_index("metodo").loc[list(METODOS)] for arq, *_ in conjuntos]

fig, eixos = plt.subplots(1, 3, figsize=(6.4, 2.3), sharey=True)
y = np.arange(len(METODOS))[::-1]
for ax, (titulo, col) in zip(eixos, [("Precisão", "precisao"), ("Recall", "recall"), ("F1", "f1")]):
    for tab, (_, nome, cor, marcador, preench, desloc) in zip(tabelas, conjuntos):
        for yi, (_, linha) in zip(y, tab.iterrows()):
            lo, hi = ler_ic(linha[f"{col}_ic"])
            ax.plot([lo, hi], [yi + desloc] * 2, color=cor, lw=1.6, solid_capstyle="round")
            ax.plot(linha[col], yi + desloc, marker=marcador, ms=6.5, mfc=preench, mec=cor, mew=1.6,
                    color=cor, label=nome if yi == y[0] else None)
    for yi, (_, linha) in zip(y, tabelas[0].iterrows()):
        ax.text(linha[col], yi + 0.33, virgula(linha[col]), ha="center", va="bottom", fontsize=7.5)
    ax.set_title(titulo, fontsize=9, pad=4)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "0,25", "0,50", "0,75", "1"])
    ax.set_ylim(-0.6, len(METODOS) - 0.25)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
eixos[0].set_yticks(y)
eixos[0].set_yticklabels(METODOS.values())
handles, labels = eixos[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.06), fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("figuras/metricas_ic.pdf", bbox_inches="tight")

# --- figura 2: matriz de confusão LLM x rótulo final ----------------------------
CATS = ["misoginia", "neutro", "odio_geral", "pular"]
NOMES = ["Misoginia", "Neutro", "Ódio geral", "Pular"]
finais = pd.read_csv("dados/processados/rotulos_finais.csv")[["id", "rotulo_final"]]
llm = pd.read_csv("resultados/llm_rotulos.csv")[["id", "rotulo"]]
comp = finais.merge(llm, on="id").dropna(subset=["rotulo"])
comp = comp[comp["rotulo_final"] != "empate"]
M = pd.crosstab(comp["rotulo_final"], comp["rotulo"]).reindex(index=CATS, columns=CATS, fill_value=0).values
porc = M / M.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(3.9, 3.1))
ax.imshow(porc, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]}\n({virgula(porc[i, j] * 100, 0)}%)", ha="center", va="center", fontsize=7.5,
                color="white" if porc[i, j] > 0.55 else "black")
ax.set_xticks(range(4))
ax.set_xticklabels(NOMES, fontsize=7.5)
ax.set_yticks(range(4))
ax.set_yticklabels(NOMES, fontsize=7.5)
ax.set_xlabel("Rótulo atribuído pela LLM")
ax.set_ylabel("Rótulo final (anotadoras)")
ax.set_xticks(np.arange(-.5, 4), minor=True)
ax.set_yticks(np.arange(-.5, 4), minor=True)
ax.grid(which="minor", color="white", lw=2)
ax.tick_params(which="both", length=0)
for lado in ax.spines.values():
    lado.set_visible(False)
fig.tight_layout()
fig.savefig("figuras/matriz_llm.pdf", bbox_inches="tight")
print(f"Figuras salvas em figuras/ (matriz com {M.sum()} postagens)")