import json
import os
import sys

import pandas as pd

ENTRADA = "dados/processados/reddit_limpo.csv"
SAIDA_AMOSTRA = "dados/processados/amostra_rotulagem.csv"
SAIDA_FORMULARIO = "rotulagem/posts.json"

SEED = 42
MAX_PALAVRAS = 400
N = {"odio_kw": 125, "odio_outro": 75, "neutro_kw": 50, "neutro_outro": 50}
N_PILOTO = 60
KW = (r"mulher|femin|feminazi|vadia|puta|piranha|vagabunda|red ?pill|incel|hipergam|mgtow|"
      r"interesseira|gold ?digger|namorada|esposa|garota|menina|moça|f[eê]mea|macho|chad|stacy")
RE_INGLES = r"\b(?:the|and|what|with|this|you)\b"
RE_PORTUGUES = r"\b(?:que|não|para|uma|você)\b"


def sortear(d):
    low = d["texto"].str.lower()
    ingles = low.str.contains(RE_INGLES) & ~low.str.contains(RE_PORTUGUES)
    d = d[~ingles & (d["texto"].str.split().str.len() <= MAX_PALAVRAS)].copy()
    d["estrato"] = d["origem"] + "_" + low.loc[d.index].str.contains(KW).map({True: "kw", False: "outro"})
    print(f"Elegíveis: {len(d)} posts ({int(ingles.sum())} em inglês e os com mais de "
          f"{MAX_PALAVRAS} palavras ficaram de fora)")
    print(d["estrato"].value_counts().to_string(), "\n")

    amostra = pd.concat([d[d["estrato"] == e].sample(n, random_state=SEED) for e, n in N.items()])

    # piloto estratificado: a mesma proporção de cada estrato vai para o início
    amostra = amostra.sample(frac=1, random_state=SEED)
    idx_piloto = [i for _, g in amostra.groupby("estrato")
                  for i in g.index[:round(N_PILOTO * len(g) / len(amostra))]]
    piloto = amostra.loc[idx_piloto]
    resto = amostra.drop(piloto.index)
    amostra = pd.concat([piloto.sample(frac=1, random_state=SEED), resto]).reset_index(drop=True)
    amostra["ordem"] = amostra.index + 1
    amostra["piloto"] = amostra["ordem"] <= len(piloto)
    return amostra


def reconstruir_da_rotulagem(d):
    """Refaz amostra_rotulagem.csv a partir do posts.json do formulário (fonte da verdade)."""
    with open(SAIDA_FORMULARIO, encoding="utf-8") as f:
        posts = json.load(f)
    ordem = pd.DataFrame({"id": [p["id"] for p in posts], "ordem": [p["n"] for p in posts],
                          "piloto": [bool(p["p"]) for p in posts]})
    faltando = set(ordem["id"]) - set(d["id"])
    if faltando:
        raise SystemExit(f"{len(faltando)} posts do formulário não estão no reddit_limpo.csv: "
                         "confira se o 01 foi rodado com os mesmos CSVs do Apify.")
    low = d["texto"].str.lower()
    d = d.assign(estrato=d["origem"] + "_" + low.str.contains(KW).map({True: "kw", False: "outro"}))
    return ordem.merge(d, on="id", how="left")[list(d.columns) + ["ordem", "piloto"]]


if __name__ == "__main__":
    d = pd.read_csv(ENTRADA)

    # A amostra rotulada é a que está no formulário (rotulagem/posts.json). Ela foi sorteada em
    # 28/09/2026; o resultado do sorteio depende da ordem em que os CSVs do Apify são lidos, que
    # varia com o nome dos arquivos. Por isso, se o posts.json existir, a amostra é reconstruída
    # a partir dele, e não sorteada de novo.
    if os.path.exists(SAIDA_FORMULARIO) and "--sortear-novamente" not in sys.argv:
        amostra = reconstruir_da_rotulagem(d)
        amostra.to_csv(SAIDA_AMOSTRA, index=False, encoding="utf-8-sig")
        print(amostra.groupby(["piloto", "estrato"]).size().to_string(), "\n")
        print(f"Amostra reconstruída a partir de {SAIDA_FORMULARIO} ({len(amostra)} posts, mesma ordem "
              f"do formulário) e salva em {SAIDA_AMOSTRA}.")
        sys.exit(0)

    # sorteio novo (só quando não há rotulagem): ordena por id para não depender da ordem dos arquivos
    amostra = sortear(d.sort_values("id").reset_index(drop=True))
    print(amostra.groupby(["piloto", "estrato"]).size().to_string(), "\n")
    os.makedirs(os.path.dirname(SAIDA_AMOSTRA), exist_ok=True)
    os.makedirs(os.path.dirname(SAIDA_FORMULARIO), exist_ok=True)
    amostra.to_csv(SAIDA_AMOSTRA, index=False, encoding="utf-8-sig")
    posts = [{"n": int(r.ordem), "id": r.id, "p": bool(r.piloto), "t": r.texto} for r in amostra.itertuples()]
    with open(SAIDA_FORMULARIO, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Amostra salva em {SAIDA_AMOSTRA} e textos do formulário em {SAIDA_FORMULARIO}.")