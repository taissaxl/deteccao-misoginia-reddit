import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from lexico import carregar_lexico, termos_no_texto

DIR = "resultados"
ARQ_LIMIAR = "modelos/bertimbau-misoginia/limiar.json"
ARQ_ROTULOS = "dados/processados/rotulos_finais.csv"
ARQ_AMOSTRA = "dados/processados/amostra_rotulagem.csv"
GRADE = np.round(np.arange(0.01, 1.00, 0.01), 2)

log = []


def registrar(msg=""):
    print(msg)
    log.append(str(msg))


def cascata(p, lex, baixo, alto):
    p, lex = np.asarray(p), np.asarray(lex)
    return np.where(p >= alto, 1, np.where(p < baixo, 0, lex)).astype(int)


def metricas(y, yhat):
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {"precisao": round(float(prec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4)}


def marcar_lexico(df, lexico):
    termos = df["texto_normalizado"].fillna("").map(lambda t: termos_no_texto(t, lexico))
    df["termos_lexico"] = termos.map("; ".join)
    df["lexico"] = (termos.map(len) > 0).astype(int)
    return df


if __name__ == "__main__":
    lexico = carregar_lexico()
    limiar_bert = json.load(open(ARQ_LIMIAR))["limiar"] if os.path.exists(ARQ_LIMIAR) else 0.5
    registrar(f"Léxico: {len(lexico)} termos | limiar do BERTimbau sozinho: {limiar_bert:.2f}\n")

    # --- calibração na validação -------------------------------------------
    val = marcar_lexico(pd.read_csv(f"{DIR}/probabilidades_validacao.csv"), lexico)
    y, p, lex = val["label"].values, val["prob_normalizado"].values, val["lexico"].values
    registrar(f"Validação: {len(val)} textos, {int(y.sum())} de misoginia, {int(lex.sum())} com termo do léxico")

    melhor = None
    for baixo in GRADE:
        for alto in GRADE[GRADE > baixo]:
            f1 = f1_score(y, cascata(p, lex, baixo, alto), zero_division=0)
            chave = (round(f1, 6), -(alto - baixo), -baixo)
            if melhor is None or chave > melhor[0]:
                melhor = (chave, float(baixo), float(alto))
    _, baixo, alto = melhor

    m_bert = metricas(y, (p >= limiar_bert).astype(int))
    m_casc = metricas(y, cascata(p, lex, baixo, alto))
    m_lex = metricas(y, lex)
    na_faixa = int(((p >= baixo) & (p < alto)).sum())
    registrar(f"Limiares escolhidos: baixo = {baixo:.2f}, alto = {alto:.2f} "
              f"({na_faixa} textos da validação caem na faixa intermediária)")
    registrar(f"  BERTimbau (limiar {limiar_bert:.2f}): {m_bert}")
    registrar(f"  Cascata:                  {m_casc}")
    registrar(f"  Léxico sozinho:           {m_lex}")
    registrar("  (valores na validação, usada para escolher os limiares: não são a avaliação final)")
    json.dump({"limiar_baixo": baixo, "limiar_alto": alto, "limiar_bertimbau": limiar_bert,
               "calibrado_em": "validacao das bases publicas", "validacao_bertimbau": m_bert,
               "validacao_cascata": m_casc, "validacao_lexico": m_lex, "termos_lexico": sorted(lexico)},
              open(f"{DIR}/cascata_parametros.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # aplicação 
    for nome in ["teste", "amostra", "reddit"]:
        if nome == "amostra":
            # monta a amostra a partir das 2.547 postagens e da amostra rotulada (fonte: posts.json),
            # para não depender de um probabilidades_amostra.csv gerado com outra versão da amostra
            ids = pd.read_csv(ARQ_AMOSTRA)[["id", "ordem", "piloto", "estrato"]]
            df = ids.merge(pd.read_csv(f"{DIR}/probabilidades_reddit.csv"), on="id", how="left")
            if df["prob_normalizado"].isna().any():
                raise SystemExit("Há posts da amostra sem probabilidade em probabilidades_reddit.csv.")
        else:
            arq = f"{DIR}/probabilidades_{nome}.csv"
            if not os.path.exists(arq):
                continue
            df = pd.read_csv(arq)
        df = marcar_lexico(df, lexico)
        df["pred_bertimbau"] = (df["prob_original"] >= limiar_bert).astype(int)
        df["pred_bertimbau_norm"] = (df["prob_normalizado"] >= limiar_bert).astype(int)
        df["pred_cascata"] = cascata(df["prob_normalizado"], df["lexico"], baixo, alto)
        df["pred_lexico"] = df["lexico"]
        df.to_csv(f"{DIR}/previsoes_{nome}.csv", index=False, encoding="utf-8")
        registrar(f"\n{nome}: {len(df)} textos | léxico em {int(df['lexico'].sum())} | "
                  f"previstos como misóginos: BERTimbau {int(df['pred_bertimbau'].sum())}, "
                  f"cascata {int(df['pred_cascata'].sum())}")

        if nome == "amostra" and os.path.exists(ARQ_ROTULOS):
            rot = pd.read_csv(ARQ_ROTULOS)[["id", "rotulo_final"]]
            pil = df[df["piloto"] == True].merge(rot, on="id")  # noqa: E712
            pil = pil[pil["rotulo_final"].isin(["misoginia", "odio_geral", "neutro"])]
            yp = (pil["rotulo_final"] == "misoginia").astype(int)
            registrar(f"  Conferência no piloto ({len(pil)} postagens, {int(yp.sum())} misóginas): "
                      f"BERTimbau {metricas(yp, pil['pred_bertimbau_norm'])} | "
                      f"cascata {metricas(yp, pil['pred_cascata'])}")

    with open(f"{DIR}/relatorio_cascata.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))