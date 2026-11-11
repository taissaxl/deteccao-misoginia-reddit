import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DIR_MODELO = "modelos/bertimbau-misoginia"
DIR_RESULTADOS = "resultados"
MAX_LEN = 512
LOTE = 32

os.makedirs(DIR_RESULTADOS, exist_ok=True)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {dispositivo}")

if not os.path.isfile(os.path.join(DIR_MODELO, "config.json")):
    raise SystemExit(f"Modelo não encontrado em {DIR_MODELO}/. Rode o 05_treinar_bertimbau.py antes, "
                     "na mesma sessão do Colab (ou com o modelo salvo no Drive).")
tokenizador = AutoTokenizer.from_pretrained(DIR_MODELO)
modelo = AutoModelForSequenceClassification.from_pretrained(DIR_MODELO).to(dispositivo).eval()
log = []


def registrar(msg=""):
    print(msg)
    log.append(str(msg))


@torch.no_grad()
def probabilidades(textos):
    textos = [str(t) for t in textos]
    saida = []
    for i in range(0, len(textos), LOTE):
        enc = tokenizador(textos[i:i + LOTE], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to(dispositivo)
        logits = modelo(**enc).logits.float()
        saida.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(saida) if saida else np.array([])


def n_tokens(textos):
    return np.array([len(tokenizador(str(t), add_special_tokens=True)["input_ids"]) for t in textos])


def inferir(nome, df, colunas_saida):
    df = df.copy()
    df["prob_original"] = probabilidades(df["texto"])
    iguais = df["texto"].astype(str) == df["texto_normalizado"].astype(str)
    df["prob_normalizado"] = df["prob_original"]
    if (~iguais).any():
        df.loc[~iguais, "prob_normalizado"] = probabilidades(df.loc[~iguais, "texto_normalizado"])
    df["truncado"] = (n_tokens(df["texto_normalizado"]) > MAX_LEN).astype(int)
    df[colunas_saida + ["prob_original", "prob_normalizado", "truncado"]].to_csv(
        f"{DIR_RESULTADOS}/probabilidades_{nome}.csv", index=False, encoding="utf-8")
    registrar(f"{nome:10s}: {len(df):6d} textos | {int((~iguais).sum()):3d} com normalização | "
              f"{int(df['truncado'].sum()):4d} truncados em {MAX_LEN} tokens")
    return df


val = pd.read_csv("dados/bases/normalizadas/validacao.csv")
inferir("validacao", val, ["texto", "texto_normalizado", "rotulo", "label", "fonte"])

teste = pd.read_csv("dados/bases/normalizadas/teste.csv")
inferir("teste", teste, ["texto", "texto_normalizado", "rotulo", "label", "fonte"])

reddit = pd.read_csv("dados/processados/reddit_normalizado.csv")
cols_reddit = [c for c in ["id", "origem", "modo_coleta", "communityName", "upVotes", "upVoteRatio",
                           "numberOfComments", "texto", "texto_normalizado"] if c in reddit.columns]
reddit = inferir("reddit", reddit, cols_reddit)

amostra_ids = pd.read_csv("dados/processados/amostra_rotulagem.csv")[["id", "ordem", "piloto", "estrato"]]
amostra = amostra_ids.merge(reddit, on="id", how="left")
amostra[["id", "ordem", "piloto", "estrato", "texto", "texto_normalizado",
         "prob_original", "prob_normalizado", "truncado"]].to_csv(
    f"{DIR_RESULTADOS}/probabilidades_amostra.csv", index=False, encoding="utf-8")
registrar(f"{'amostra':10s}: {len(amostra):6d} textos | {int(amostra['truncado'].sum()):4d} truncados "
          f"(esperado 0: posts com mais de 400 palavras ficaram fora da amostra)")

with open(f"{DIR_RESULTADOS}/relatorio_inferencia.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))