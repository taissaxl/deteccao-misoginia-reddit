import os
import re
from collections import Counter

import pandas as pd

ARQ_DICIONARIO = "recursos/dicionario_normalizacao.csv"
ARQ_REDDIT = "dados/processados/reddit_limpo.csv"
DIR_BASES = "dados/bases/processadas"
DIR_BASES_SAIDA = "dados/bases/normalizadas"
SAIDA_REDDIT = "dados/processados/reddit_normalizado.csv"
SAIDA_RELATORIO = "dados/processados/relatorio_normalizacao.txt"
SAIDA_CANDIDATOS = "dados/processados/candidatos_variantes.csv"

# Detecção de candidatos: palavra com letras e dígitos que, trocando os dígitos pelas letras
# parecidas, vira uma palavra só de letras. Serve apenas para sugerir variantes novas.
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
RE_TOKEN_COM_DIGITO = re.compile(r"(?<![\w/@#])[\wÀ-ÿ@]*\d[\wÀ-ÿ@]*(?![\w/])")
# unidades e padrões comuns que não são grafias alteradas (30kg, 10min, 140bpm, 13th...)
RE_MEDIDA = re.compile(r"^\d+(kg|g|grs|mg|ml|l|km|m|cm|mm|min|seg|s|h|hr|hrs|hs|ms|bpm|kcal|mil|mi|k|kk|"
                       r"am|pm|th|st|nd|rd|anos|ano|x|vez|vezes|gb|mb|kb|tb|hz)$")

log = []


def registrar(msg=""):
    print(msg)
    log.append(str(msg))


def carregar_dicionario():
    d = pd.read_csv(ARQ_DICIONARIO, dtype=str).fillna("")
    d["variante"] = d["variante"].str.strip().str.lower()
    d["forma_canonica"] = d["forma_canonica"].str.strip()
    duplicadas = d["variante"][d["variante"].duplicated()].tolist()
    if duplicadas:
        raise ValueError(f"Variantes repetidas no dicionário: {duplicadas}")
    mapa = dict(zip(d["variante"], d["forma_canonica"]))
    # variantes mais longas primeiro, para "p0rr4444" não ser casada como "p0rr4"
    alternativas = sorted(mapa, key=len, reverse=True)
    padrao = re.compile(r"(?<![\w@])(" + "|".join(re.escape(v) for v in alternativas) + r")(?![\w@])",
                        flags=re.IGNORECASE)
    return mapa, padrao


def ajustar_caixa(original, canonica):
    letras = [c for c in original if c.isalpha()]
    if letras and all(c.isupper() for c in letras):
        return canonica.upper()
    if letras and letras[0].isupper():
        return canonica[:1].upper() + canonica[1:]
    return canonica


def normalizar_serie(textos, mapa, padrao):
    contagem = Counter()
    normalizados, n_subst = [], []

    def trocar(m):
        orig = m.group(0)
        contagem[orig.lower()] += 1
        return ajustar_caixa(orig, mapa[orig.lower()])

    for t in textos.fillna("").astype(str):
        antes = sum(contagem.values())
        normalizados.append(padrao.sub(trocar, t))
        n_subst.append(sum(contagem.values()) - antes)
    return pd.Series(normalizados, index=textos.index), pd.Series(n_subst, index=textos.index), contagem


def candidatos(textos, mapa, nome_base):
    achados = Counter()
    exemplo = {}
    for t in textos.fillna("").astype(str):
        for m in RE_TOKEN_COM_DIGITO.findall(t):
            w = m.lower()
            if w in mapa or RE_MEDIDA.match(w) or len(w) < 4:
                continue
            if len(re.findall(r"[a-zà-ÿ]", w)) < 2:
                continue
            if not re.fullmatch(r"[a-zà-ÿ]+", w.translate(LEET)):
                continue
            achados[w] += 1
            if w not in exemplo:
                i = t.lower().find(w)
                exemplo[w] = t[max(0, i - 50): i + len(w) + 50].replace("\n", " ")
    return [{"candidato": w, "frequencia": n, "base": nome_base, "exemplo": exemplo[w]}
            for w, n in achados.items()]


def processar(nome, df, mapa, padrao, todos_candidatos):
    df = df.copy()
    df["texto_normalizado"], df["n_substituicoes"], contagem = normalizar_serie(df["texto"], mapa, padrao)
    total = int(df["n_substituicoes"].sum())
    textos_afetados = int((df["n_substituicoes"] > 0).sum())
    registrar(f"{nome:10s}: {len(df):6d} textos | {total:4d} substituições em {textos_afetados:4d} textos "
              f"({textos_afetados / max(len(df), 1):.2%})")
    if contagem:
        registrar("            " + ", ".join(f"{v} -> {mapa[v]} ({n})" for v, n in contagem.most_common()))
    todos_candidatos.extend(candidatos(df["texto"], mapa, nome))
    return df, total, textos_afetados


if __name__ == "__main__":
    mapa, padrao = carregar_dicionario()
    registrar(f"Dicionário: {len(mapa)} variantes ({ARQ_DICIONARIO})\n")
    todos_candidatos = []

    reddit = pd.read_csv(ARQ_REDDIT)
    reddit, sub_reddit, _ = processar("reddit", reddit, mapa, padrao, todos_candidatos)
    reddit.to_csv(SAIDA_REDDIT, index=False, encoding="utf-8-sig")

    os.makedirs(DIR_BASES_SAIDA, exist_ok=True)
    sub_publicas, n_publicas = 0, 0
    for parte in ["treino", "validacao", "teste"]:
        df = pd.read_csv(f"{DIR_BASES}/{parte}.csv")
        df, total, _ = processar(parte, df, mapa, padrao, todos_candidatos)
        df.to_csv(f"{DIR_BASES_SAIDA}/{parte}.csv", index=False, encoding="utf-8")
        sub_publicas += total
        n_publicas += len(df)

    registrar(f"\nBases públicas: {sub_publicas} substituições em {n_publicas} textos "
              f"({sub_publicas / n_publicas:.3%} dos textos, no máximo).")

    cand = pd.DataFrame(todos_candidatos, columns=["candidato", "frequencia", "base", "exemplo"])
    if len(cand):
        cand = (cand.groupby("candidato")
                    .agg(frequencia=("frequencia", "sum"), bases=("base", lambda s: ", ".join(sorted(set(s)))),
                         exemplo=("exemplo", "first"))
                    .sort_values("frequencia", ascending=False).reset_index())
    cand.to_csv(SAIDA_CANDIDATOS, index=False, encoding="utf-8-sig")
    registrar(f"\n{len(cand)} candidatos fora do dicionário salvos em {SAIDA_CANDIDATOS} (revisão manual).")

    with open(SAIDA_RELATORIO, "w", encoding="utf-8") as f:
        f.write("\n".join(log))