import os
 
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
 
SEMENTE = 42
N_BOOT = 1000
REGRA_EMPATE = "excluir"
CATEGORIAS = ["misoginia", "odio_geral", "neutro", "pular"]
VALIDOS = ["misoginia", "odio_geral", "neutro"]
 
ARQ_ROTULOS = "dados/processados/rotulos_misoginia_reddit.csv"
ARQ_AMOSTRA = "dados/processados/amostra_rotulagem.csv"
ARQ_PREV = "resultados/previsoes_amostra.csv"
ARQ_LLM = "resultados/llm_rotulos.csv"
DIR = "resultados"
 
METODOS = {
    "BERTimbau": "pred_bertimbau",
    "BERTimbau (normalizado)": "pred_bertimbau_norm",
    "Cascata BERTimbau + léxico": "pred_cascata",
    "Léxico sozinho": "pred_lexico",
    "LLM (Gemini 3.5 Flash Lite)": "pred_llm",
}
REFERENCIA = "BERTimbau (normalizado)"
 
log = []
 
 
def registrar(msg=""):
    print(msg)
    log.append(str(msg))
 
 
def fleiss_kappa(tabela):
    """tabela: itens x categorias com a contagem de votos; todos os itens com o mesmo nº de votos."""
    t = np.asarray(tabela, dtype=float)
    n = t.sum(axis=1)[0]
    if len(t) == 0 or not np.all(t.sum(axis=1) == n):
        return float("nan")
    p_j = t.sum(axis=0) / t.sum()
    p_i = ((t ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")
 
 
def tabela_votos(largo, anotadoras, binario=False):
    votos = largo[anotadoras]
    if binario:
        votos = votos.eq("misoginia")
        return np.column_stack([votos.sum(axis=1), (~votos).sum(axis=1)])
    return np.column_stack([votos.eq(c).sum(axis=1) for c in CATEGORIAS])
 
 
def interpretar(k):
    """Escala de Landis e Koch (1977)."""
    if np.isnan(k):
        return "-"
    for lim, txt in [(0, "pobre"), (0.2, "leve"), (0.4, "razoável"), (0.6, "moderada"),
                     (0.8, "substancial"), (1.01, "quase perfeita")]:
        if k <= lim:
            return txt
    return "quase perfeita"
 
 
def maioria(votos):
    cont = pd.Series(votos).value_counts()
    return (cont.index[0], int(cont.iloc[0])) if cont.iloc[0] >= 2 else ("empate", 1)
 
 
def tipo_final(tipos):
    tipos = [t for t in tipos if t in ("explicita", "implicita")]
    if not tipos:
        return None
    cont = pd.Series(tipos).value_counts()
    if len(cont) > 1 and cont.iloc[0] == cont.iloc[1]:
        return "explicita"
    return cont.index[0]
 
 
def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
 
 
def avaliar(finais, prev, incluir_piloto, titulo, sufixo):
    base = finais if incluir_piloto else finais[~finais["piloto"]]
    av = base[base["rotulo_final"].isin(VALIDOS)][["id", "rotulo_final", "tipo_final"]]
    av = av.merge(prev, on="id", how="left")
    av["y"] = (av["rotulo_final"] == "misoginia").astype(int)
    excl = base[~base["rotulo_final"].isin(VALIDOS)]["rotulo_final"].value_counts().to_dict()
    registrar(f"\n{titulo}: {len(av)} postagens, {int(av['y'].sum())} misóginas "
              f"(fora da avaliação: {excl or 'nenhuma'})")
 
    rng = np.random.default_rng(SEMENTE)
    amostras = [rng.integers(0, len(av), len(av)) for _ in range(N_BOOT)]
    y_all = av["y"].values
    boot_f1 = {}
    linhas = []
    for nome, col in METODOS.items():
        ok = av[col].notna().values
        y, p = y_all[ok], av[col].values[ok].astype(int)
        prec, rec, f1 = prf(y, p)
        bs = []
        for idx in amostras:
            idx = idx[ok[idx]]
            bs.append(prf(y_all[idx], av[col].values[idx].astype(int)))
        bs = np.array(bs)
        boot_f1[nome] = bs[:, 2]
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        linhas.append({"metodo": nome, "n": int(ok.sum()), "misoginas": int(y.sum()), "previstas": int(p.sum()),
                       "vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn),
                       "precisao": round(prec, 3), "precisao_ic": f"[{lo[0]:.3f}; {hi[0]:.3f}]",
                       "recall": round(rec, 3), "recall_ic": f"[{lo[1]:.3f}; {hi[1]:.3f}]",
                       "f1": round(f1, 3), "f1_ic": f"[{lo[2]:.3f}; {hi[2]:.3f}]"})
    met = pd.DataFrame(linhas)
    for i, r in met.iterrows():
        if r["metodo"] != REFERENCIA:
            d = boot_f1[r["metodo"]] - boot_f1[REFERENCIA]
            lo, hi = np.percentile(d, [2.5, 97.5])
            met.loc[i, "dif_f1_vs_ref"] = round(float(d.mean()), 3)
            met.loc[i, "dif_f1_ic"] = f"[{lo:.3f}; {hi:.3f}]"
    met.to_csv(f"{DIR}/avaliacao_metricas{sufixo}.csv", index=False, encoding="utf-8-sig")
    registrar(met[["metodo", "n", "previstas", "vp", "fp", "fn", "precisao", "precisao_ic",
                   "recall", "recall_ic", "f1", "f1_ic"]].to_string(index=False))
    if "dif_f1_vs_ref" in met:
        registrar(f"\n  Diferença de F1 para o {REFERENCIA} (IC 95% pareado; se contém 0, a diferença não é conclusiva):")
        for _, r in met.dropna(subset=["dif_f1_vs_ref"]).iterrows():
            registrar(f"    {r['metodo']}: {r['dif_f1_vs_ref']:+.3f} {r['dif_f1_ic']}")
 
    for col, nome in [("prob_original", "texto original"), ("prob_normalizado", "texto normalizado")]:
        if av["y"].nunique() == 2:
            registrar(f"  AUC do BERTimbau ({nome}): {roc_auc_score(av['y'], av[col]):.3f}")
 
    if av["tipo_final"].notna().any():
        registrar("\n  Recall por tipo de misoginia (rótulo final):")
        for tipo in ["explicita", "implicita"]:
            s = av[av["tipo_final"] == tipo]
            if len(s):
                txt = " | ".join(f"{n}: {int(s[c].fillna(0).sum())}/{len(s)}" for n, c in METODOS.items())
                registrar(f"    {tipo} (n={len(s)}): {txt}")
 
    # erros para análise qualitativa
    erros = []
    for nome, col in METODOS.items():
        s = av[av[col].notna() & (av[col] != av["y"])]
        for _, r in s.iterrows():
            erros.append({"metodo": nome, "erro": "falso positivo" if r[col] == 1 else "falso negativo",
                          "id": r["id"], "rotulo_final": r["rotulo_final"], "tipo_final": r["tipo_final"],
                          "prob_normalizado": round(r["prob_normalizado"], 4), "termos_lexico": r["termos_lexico"],
                          "rotulo_llm": r.get("rotulo_llm"), "justificativa_llm": r.get("justificativa_llm"),
                          "texto": r["texto"]})
    pd.DataFrame(erros).to_csv(f"{DIR}/avaliacao_erros{sufixo}.csv", index=False, encoding="utf-8-sig")
 
 
if __name__ == "__main__":
    rot = pd.read_csv(ARQ_ROTULOS)
    rot["rotulo"] = rot["rotulo"].str.strip()
    amostra = pd.read_csv(ARQ_AMOSTRA)[["id", "ordem", "piloto", "estrato"]]
    anotadoras = sorted(rot["anotadora"].unique())
 
    # --- 0. conferência --------------------------------------------------------
    duplic = rot.duplicated(["post_id", "anotadora"], keep=False)
    if duplic.any():
        raise SystemExit(f"Há {duplic.sum()} respostas duplicadas (mesma anotadora e postagem).")
    largo = rot.pivot(index="post_id", columns="anotadora", values="rotulo")
    largo = amostra.set_index("id").join(largo, how="left")
    faltando = largo[anotadoras].isna()
    registrar(f"Anotadoras: {', '.join(anotadoras)} | postagens: {len(largo)}")
    for a in anotadoras:
        if faltando[a].any():
            ids = largo.index[faltando[a]]
            registrar(f"  ATENÇÃO: {a} não rotulou {len(ids)} postagem(ns): "
                      + ", ".join(f"{i} (nº {int(largo.loc[i, 'ordem'])})" for i in ids))
    completas = largo[~faltando.any(axis=1)]
    if len(completas) < len(largo):
        registrar(f"  Concordância calculada sobre as {len(completas)} postagens com as três respostas.")
 
    # --- 1. concordância --------------------------------------------------------
    registrar("\n1. CONCORDÂNCIA ENTRE AS ANOTADORAS (Kappa de Fleiss; interpretação de Landis e Koch)")
    conc = []
    for nome, filtro in [("Piloto", completas["piloto"] == True),          # noqa: E712
                         ("Avaliação", completas["piloto"] == False),      # noqa: E712
                         ("Total", completas["piloto"].notna())]:
        sub = completas[filtro]
        k4 = fleiss_kappa(tabela_votos(sub, anotadoras))
        k2 = fleiss_kappa(tabela_votos(sub, anotadoras, binario=True))
        unanimes = int((sub[anotadoras].nunique(axis=1) == 1).sum())
        tres = int((sub[anotadoras].nunique(axis=1) == 3).sum())
        conc.append({"conjunto": nome, "postagens": len(sub), "kappa_4_categorias": round(k4, 3),
                     "kappa_binario": round(k2, 3), "unanimes": unanimes, "empates_triplos": tres})
        registrar(f"  {nome:10s} n={len(sub):3d} | 4 categorias: {k4:.3f} ({interpretar(k4)}) | "
                  f"misoginia x resto: {k2:.3f} ({interpretar(k2)}) | unânimes: {unanimes} | "
                  f"2x1: {len(sub) - unanimes - tres} | empates triplos: {tres}")
 
    pares = []
    for i, a in enumerate(anotadoras):
        for b in anotadoras[i + 1:]:
            k = cohen_kappa_score(completas[a], completas[b], labels=CATEGORIAS)
            pares.append(k)
            registrar(f"  Cohen {a} x {b}: {k:.3f}")
 
    # --- 2. rótulos finais --------------------------------------------------------
    tipos = rot.pivot(index="post_id", columns="anotadora", values="tipo")
    finais = []
    for pid, r in largo.iterrows():
        votos = [r[a] for a in anotadoras if pd.notna(r[a])]
        rf, nv = maioria(votos)
        tf = tipo_final([tipos.loc[pid, a] for a in anotadoras
                         if r.get(a) == "misoginia" and pid in tipos.index]) if rf == "misoginia" else None
        finais.append({"id": pid, "ordem": int(r["ordem"]), "piloto": bool(r["piloto"]),
                       "estrato": r["estrato"], "rotulo_final": rf, "tipo_final": tf,
                       "votos_maioria": nv, "n_respostas": len(votos), "unanime": nv == len(votos) == 3,
                       **{f"rotulo_{a}": r[a] for a in anotadoras}})
    finais = pd.DataFrame(finais)
    finais.to_csv("dados/processados/rotulos_finais.csv", index=False, encoding="utf-8-sig")
 
    registrar("\n2. RÓTULOS FINAIS (voto majoritário)")
    tab = pd.crosstab(finais["rotulo_final"], finais["piloto"].map({True: "piloto", False: "avaliação"}),
                      margins=True, margins_name="total")
    registrar(tab.to_string())
    mis = finais[finais["rotulo_final"] == "misoginia"]
    registrar(f"  Tipo da misoginia: {mis['tipo_final'].value_counts().to_dict()}")
    origem = finais.merge(pd.read_csv(ARQ_AMOSTRA)[["id", "modo_coleta", "communityName"]], on="id")
    for col, nome in [("estrato", "estrato"), ("modo_coleta", "modo de coleta")]:
        registrar(f"\n  Rótulos finais por {nome}:")
        registrar(pd.crosstab(origem[col], origem["rotulo_final"], margins=True, margins_name="total").to_string())
    registrar("\n  Subreddits das postagens misóginas: "
              + str(origem.loc[origem["rotulo_final"] == "misoginia", "communityName"].value_counts().to_dict()))
    registrar("  Subreddits mais frequentes na amostra: "
              + str(origem["communityName"].value_counts().head(10).to_dict()))
    pul = origem[origem["rotulo_final"] == "pular"].merge(
        pd.read_csv(ARQ_AMOSTRA)[["id", "texto", "termos_busca"]], on="id")
    registrar(f"  Postagens puladas por modo de coleta: {pul['modo_coleta'].value_counts().to_dict()} "
              f"(lista em {DIR}/postagens_puladas.csv, para inspeção manual)")
    pul.assign(inicio_texto=pul["texto"].str.slice(0, 200).str.replace("\n", " "), idioma_inspecao="",
               motivo_inspecao="")[["id", "ordem", "modo_coleta", "communityName", "termos_busca", "inicio_texto",
                                    "idioma_inspecao", "motivo_inspecao"]] \
        .sort_values(["modo_coleta", "communityName"]) \
        .to_csv(f"{DIR}/postagens_puladas.csv", index=False, encoding="utf-8-sig")
    origem.groupby(["estrato", "modo_coleta", "communityName", "rotulo_final"]).size().rename("postagens") \
        .reset_index().to_csv(f"{DIR}/rotulos_por_origem.csv", index=False, encoding="utf-8-sig")
    if (finais["rotulo_final"] == "empate").any():
        registrar(f"  Empates triplos: {int((finais['rotulo_final'] == 'empate').sum())} "
                  f"(regra: {REGRA_EMPATE})")
 
    # --- 3. LLM como anotadora ----------------------------------------------------
    prev = pd.read_csv(ARQ_PREV)
    tem_llm = os.path.exists(ARQ_LLM)
    if tem_llm:
        llm = pd.read_csv(ARQ_LLM)[["id", "status", "rotulo", "tipo", "trecho", "justificativa"]]
        llm = llm.rename(columns={"rotulo": "rotulo_llm", "tipo": "tipo_llm", "trecho": "trecho_llm",
                                  "justificativa": "justificativa_llm", "status": "status_llm"})
        prev = prev.merge(llm, on="id", how="left")
        prev["pred_llm"] = np.where(prev["rotulo_llm"].isna(), np.nan,
                                    (prev["rotulo_llm"] == "misoginia").astype(float))
 
        registrar("\n3. LLM COMO QUARTA ANOTADORA")
        comp = finais.merge(llm, on="id")
        comp = comp[comp["rotulo_llm"].notna() & (comp["rotulo_final"] != "empate")]
        registrar(f"  Postagens comparadas: {len(comp)} (fora: bloqueadas pela LLM e empates)")
        k4 = cohen_kappa_score(comp["rotulo_final"], comp["rotulo_llm"], labels=CATEGORIAS)
        k2 = cohen_kappa_score(comp["rotulo_final"] == "misoginia", comp["rotulo_llm"] == "misoginia")
        registrar(f"  Cohen LLM x rótulo final: 4 categorias {k4:.3f} ({interpretar(k4)}) | "
                  f"misoginia x resto {k2:.3f} ({interpretar(k2)})")
        for a in anotadoras:
            c = comp[comp[f"rotulo_{a}"].notna()]
            registrar(f"  Cohen LLM x {a}: {cohen_kappa_score(c[f'rotulo_{a}'], c['rotulo_llm'], labels=CATEGORIAS):.3f}")
        registrar(f"  (média dos Kappas de Cohen entre pares de anotadoras humanas: {np.mean(pares):.3f})")
        com_llm = completas.join(llm.set_index("id")["rotulo_llm"], how="inner").dropna(subset=["rotulo_llm"])
        k_sem = fleiss_kappa(tabela_votos(com_llm, anotadoras))
        k_com = fleiss_kappa(tabela_votos(com_llm, anotadoras + ["rotulo_llm"]))
        registrar(f"  Fleiss nas mesmas {len(com_llm)} postagens: 3 humanas {k_sem:.3f} | 3 humanas + LLM {k_com:.3f}")
        registrar("  Matriz de confusão (linhas: rótulo final; colunas: LLM):")
        registrar(pd.crosstab(comp["rotulo_final"], comp["rotulo_llm"]).to_string())
    else:
        registrar(f"\n3. LLM: {ARQ_LLM} não encontrado; a LLM fica fora da avaliação.")
        METODOS.pop("LLM (Gemini 3.5 Flash Lite)")
 
    # --- 4. avaliação ---------------------------------------------------------------
    avaliar(finais, prev, False, "4. AVALIAÇÃO DOS CLASSIFICADORES (postagens de avaliação)", "")
    # complementar: as 300 postagens. O piloto não foi usado para ajustar nenhum classificador
    # (a cascata foi calibrada na validação pública), então pode entrar como análise de sensibilidade.
    avaliar(finais, prev, True, "5. ANÁLISE COMPLEMENTAR (piloto + avaliação, 300 postagens)", "_300")
 
    pd.DataFrame(conc).to_csv(f"{DIR}/concordancia.csv", index=False, encoding="utf-8-sig")
 
    with open(f"{DIR}/relatorio_avaliacao.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))