import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DIR_TUPYE = "dados/bases/TuPyE"
DIR_MINA = "dados/bases/MINA-BR"
SAIDA = "dados/bases/processadas"
os.makedirs(SAIDA, exist_ok=True)

log = []


def registrar(msg=""):
    print(msg)
    log.append(str(msg))


RE_URL = re.compile(r"https?://\S+|www\.\S+|https?:?/?/?\S*…")
RE_USUARIO = re.compile(r"(?<!\w)@\w+|(?<![\w/])/?u/[A-Za-z0-9_-]{3,}")


def limpar(txt):
    """Mesma anonimização dos posts do Reddit: menções viram [usuario], links saem."""
    txt = RE_URL.sub(" ", str(txt))
    txt = RE_USUARIO.sub("[usuario]", txt)
    txt = re.sub(r"^rt\s+", "", txt, flags=re.IGNORECASE)          # retweet
    txt = re.sub(r"(\[usuario\]\s*)+", "[usuario] ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def chave(txt):
    return re.sub(r"[^\w\s]", "", txt.lower()).strip()


# ---------------------------------------------------------------------------
# TuPy-E
# ---------------------------------------------------------------------------
CATEGORIAS_ODIO = ["ageism", "aporophobia", "body_shame", "capacitism", "lgbtphobia", "political",
                   "racism", "religious_intolerance", "xenophobia", "other"]


def carregar_tupye(arquivo, particao):
    df = pd.read_csv(arquivo)
    ofensa = (df["aggressive"] == 1) | (df["hate"] == 1) | (df[CATEGORIAS_ODIO].sum(axis=1) > 0)
    out = pd.DataFrame({"texto": df["text"], "fonte": "TuPy-E / " + df["researcher"], "particao": particao})
    out["rotulo"] = "neutro"
    out.loc[ofensa, "rotulo"] = "odio_geral"
    out.loc[df["misogyny"] == 1, "rotulo"] = "misoginia"
    return out


registrar("== TuPy-E ==")
treino_tupy = carregar_tupye(f"{DIR_TUPYE}/multilabel/multilabel_train.csv", "treino")
teste_tupy = carregar_tupye(f"{DIR_TUPYE}/multilabel/multilabel_test.csv", "teste")
registrar(f"  treino oficial {len(treino_tupy)} | teste oficial {len(teste_tupy)}")

# ---------------------------------------------------------------------------
# MINA-BR 
# ---------------------------------------------------------------------------
# Critério de ódio no MINA-BR. A coluna 'odio' da base só vale 1 quando as TRÊS anotadoras
# concordam (unanimidade). Para ficar coerente com o TuPy-E e com a rotulagem do Reddit,
# que usam voto da maioria, o padrão aqui é maioria (2 de 3). Troque para "unanimidade"
# se quiser a classificação original da base.
CRITERIO_MINA = "maioria"


def carregar_mina():
    arquivo = f"{DIR_MINA}/MINA-BR Rotulada.csv"
    registrar("\n== MINA-BR ==")
    if not os.path.exists(arquivo):
        registrar(f"  {arquivo} não encontrado: base não incluída")
        return None
    df = pd.read_csv(arquivo)
    votos_odio = df[["disc_clas_1", "disc_clas_2", "disc_clas_3"]].fillna(0).sum(axis=1)
    votos_ofensa = df[["ofens_clas_1", "ofens_clas_2", "ofens_clas_3"]].fillna(0).sum(axis=1)
    odio = (df["odio"] == 1) if CRITERIO_MINA == "unanimidade" else (votos_odio >= 2)
    ofensivo = votos_ofensa >= 2
    out = pd.DataFrame({"texto": df["comment"].str.replace("@NOME", "@usuario", regex=False),
                        "fonte": "MINA-BR", "particao": "treino"})
    # Todos os comentários do MINA-BR são dirigidos a mulheres: ódio contra a mulher = misoginia.
    # Ofensivo sem ódio fica como odio_geral (ofensivo, não misógino), seguindo a própria anotação.
    out["rotulo"] = "neutro"
    out.loc[ofensivo, "rotulo"] = "odio_geral"
    out.loc[odio, "rotulo"] = "misoginia"
    registrar(f"  {len(df)} comentários rotulados; critério de ódio = {CRITERIO_MINA} "
              f"-> {int(odio.sum())} de misoginia")
    return out


mina = carregar_mina()

# ---------------------------------------------------------------------------
# Limpeza, duplicatas e divisão
# ---------------------------------------------------------------------------
base = pd.concat([p for p in (treino_tupy, teste_tupy, mina) if p is not None], ignore_index=True)
base["texto"] = base["texto"].map(limpar)
base = base[base["texto"].str.split().str.len() >= 3]
base["_k"] = base["texto"].map(chave)
base["label"] = (base["rotulo"] == "misoginia").astype(int)
n0 = len(base)

# 1) o teste oficial manda: texto que aparece no teste sai do treino (vazamento)
k_teste = set(base.loc[base["particao"] == "teste", "_k"])
vazou = (base["particao"] == "treino") & base["_k"].isin(k_teste)
base = base[~vazou]
# 2) duplicatas dentro de cada partição: fica a versão com rótulo de misoginia, se houver
base = base.sort_values(["label", "fonte", "texto"], ascending=[False, True, True], kind="mergesort").drop_duplicates(["particao", "_k"])
registrar(f"\nLinhas: {n0} -> {len(base)} ({vazou.sum()} do treino também estavam no teste; "
          f"{n0 - vazou.sum() - len(base)} duplicatas)")

teste = base[base["particao"] == "teste"]
treino_total = base[base["particao"] == "treino"]
treino, validacao = train_test_split(treino_total, test_size=0.1, random_state=SEED,
                                     stratify=treino_total["rotulo"] + "|" + treino_total["fonte"])

registrar("\nClasses por fonte (todas as partições):")
registrar(pd.crosstab(base["fonte"], base["rotulo"], margins=True).to_string())
for nome, parte in [("Treino", treino), ("Validação", validacao), ("Teste", teste)]:
    registrar(f"\n{nome}: {len(parte)} textos, misoginia = {parte['label'].sum()} ({parte['label'].mean():.1%})")
registrar("\nA misoginia é minoritária: use pesos por classe na loss e avalie pelo F1 da classe misoginia.")
registrar("Os 300 posts rotulados do Reddit ficam como um segundo teste, fora desta divisão.")

colunas = ["texto", "rotulo", "label", "fonte"]
treino[colunas].to_csv(f"{SAIDA}/treino.csv", index=False, encoding="utf-8")
validacao[colunas].to_csv(f"{SAIDA}/validacao.csv", index=False, encoding="utf-8")
teste[colunas].to_csv(f"{SAIDA}/teste.csv", index=False, encoding="utf-8")
with open(f"{SAIDA}/relatorio_bases.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log))
