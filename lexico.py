import re
import unicodedata

import pandas as pd

ARQ_LEXICO = "recursos/lexico_machosfera.csv"


def sem_acento(txt):
    return unicodedata.normalize("NFD", str(txt).lower()).encode("ascii", "ignore").decode()


def _padrao(termo):
    partes = [re.escape(p) for p in sem_acento(termo).split()]
    corpo = r"\s+".join(partes)
    return re.compile(r"(?<!\w)" + corpo + r"(?:s|es)?(?!\w)")


def carregar_lexico(arquivo=ARQ_LEXICO, so_incluidos=True):
    """Devolve {termo: padrão compilado}."""
    lex = pd.read_csv(arquivo, dtype=str).fillna("")
    if so_incluidos:
        lex = lex[lex["incluir"].str.strip() == "1"]
    return {t: _padrao(t) for t in lex["termo"].str.strip()}


def termos_no_texto(texto, lexico):
    t = sem_acento(texto)
    return [termo for termo, p in lexico.items() if p.search(t)]


def contem_lexico(texto, lexico):
    t = sem_acento(texto)
    return any(p.search(t) for p in lexico.values())


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ARQ_REDDIT = "dados/processados/reddit_normalizado.csv"
    ARQ_AMOSTRA = "dados/processados/amostra_rotulagem.csv"
    DIR_BASES = "dados/bases/normalizadas"
    log = []

    def registrar(msg=""):
        print(msg)
        log.append(str(msg))

    tabela = pd.read_csv(ARQ_LEXICO, dtype=str).fillna("")
    todos = carregar_lexico(so_incluidos=False)

    reddit = pd.read_csv(ARQ_REDDIT)
    ids_amostra = pd.read_csv(ARQ_AMOSTRA)[["id", "piloto"]]
    reddit = reddit.merge(ids_amostra, on="id", how="left")
    publicas = pd.concat([pd.read_csv(f"{DIR_BASES}/{p}.csv") for p in ["treino", "validacao", "teste"]],
                         ignore_index=True)

    t_reddit = reddit["texto_normalizado"].map(sem_acento)
    t_pub = publicas["texto_normalizado"].map(sem_acento)
    na_amostra = reddit["piloto"].notna()
    no_piloto = reddit["piloto"] == True  # noqa: E712

    linhas = []
    for _, r in tabela.iterrows():
        p = todos[r["termo"].strip()]
        em_reddit = t_reddit.str.contains(p)
        em_pub = t_pub.str.contains(p)
        n_pub = int(em_pub.sum())
        linhas.append({
            "termo": r["termo"], "categoria": r["categoria"], "incluir": r["incluir"],
            "reddit_posts": int(em_reddit.sum()),
            "amostra_posts": int((em_reddit & na_amostra).sum()),
            "piloto_posts": int((em_reddit & no_piloto).sum()),
            "publicas_textos": n_pub,
            "publicas_pct_misoginia": round(100 * publicas.loc[em_pub, "label"].mean(), 1) if n_pub else None,
        })
    diag = pd.DataFrame(linhas).sort_values(["categoria", "reddit_posts"], ascending=[False, False])
    diag.to_csv("dados/processados/diagnostico_lexico.csv", index=False, encoding="utf-8-sig")

    incluidos = carregar_lexico()
    flag_reddit = reddit["texto_normalizado"].map(lambda x: contem_lexico(x, incluidos))
    flag_pub = publicas["texto_normalizado"].map(lambda x: contem_lexico(x, incluidos))
    registrar(f"Léxico: {len(tabela)} termos, {len(incluidos)} incluídos (específicos)\n")
    registrar(f"Reddit (2.547): {int(flag_reddit.sum())} postagens com termo do léxico")
    registrar(f"  amostra (300): {int((flag_reddit & na_amostra).sum())} | piloto (60): {int((flag_reddit & no_piloto).sum())}")
    registrar(f"Bases públicas: {int(flag_pub.sum())} textos com termo do léxico; "
              f"{100 * publicas.loc[flag_pub, 'label'].mean():.1f}% deles rotulados como misoginia "
              f"(taxa geral: {100 * publicas['label'].mean():.1f}%)")
    registrar("\nPor termo: dados/processados/diagnostico_lexico.csv")
    with open("dados/processados/relatorio_lexico.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))