import glob
import hashlib
import html
import os
import re
import secrets
import unicodedata
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DIR_NEUTROS = "dados/brutos/neutros"
DIR_ODIO = "dados/brutos/odio"
DIR_SAIDA = "dados/processados"
 
MIN_PALAVRAS = 5      # mantém textos com mais de 4 palavras
EXIGIR_CORPO = True   # post só com título geralmente é foto ou link: a anotadora não consegue julgar
TIPOS = ("post",)     # para incluir comentários de uma coleta futura: ("post", "comment")
 
# Sal do pseudônimo dos autores. Fica num arquivo fora do controle de versão
# (dados/brutos/ está no .gitignore). Criado automaticamente na primeira execução;
# com o mesmo sal, o mesmo autor recebe sempre o mesmo código.
ARQUIVO_SAL = "dados/brutos/.sal_pseudonimos"
 
# Termos usados na busca do Apify para montar a coleta de ódio (7 extrações).
# NÃO são usados para filtrar nem para amostrar: servem só como variável de análise
# (ex.: proporção de misoginia entre posts com e sem esses termos).
# A busca do Reddit devolve posts relacionados mesmo sem o termo no texto, por isso
# a maioria dos posts coletados não contém nenhum deles.
TERMOS_BUSCA = [
    "modernete", "mãe solteira", "mulher independente", "macho alfa", "80/20", "depósito",
    "hipergamia", "femoid", "miqueinha", "miqueinhas", "mulher 304", "betinha", "homem beta",
    "bostileira", "femimi", "antiotário", "antiotario", "antitrouxa", "coomecrata", "coomecracia",
    "mulh3r", "mulh3res", "pílula vermelha", "pílula azul", "homem de valor", "homem de alto valor",
    "homem moderno", "mulher moderna", "vsm", "beta", "escravoceta", "mangina", "feministo", "gados",
    "blue pill", "bluepill", "mask pill", "red pill", "redpill", "blackpill", "black pill",
    "polo feminino", "polo masculino", "mimimi", "ordin4ria", "0rd1nar14", "saf4da",
    "femibostan1sm0", "fem1bostan1smo", "mulher interesseira",
    "carrossel", "msol", "wall", "bateu no wall", "looksmaxxing", "chad", "muié", "muie",
    "marmita", "feminazi",
]
 
# Modo de cada extração. Nas extrações feitas com URLs de subreddits (startUrls), o Apify
# IGNORA os termos de busca e coleta as postagens recentes dos subreddits informados.
# Só três extrações foram de fato buscas por termo; a de 07/09 está na pasta
# neutros, mas usou termos da machosfera (registrado como limitação na metodologia).
EXTRACOES_POR_BUSCA = (
    "dataset_reddit-scraper-lite_2026-09-02_19-32-09-450.csv",
    "dataset_reddit-scraper-lite_2026-09-04_19-16-56-452.csv",
    "dataset_reddit-scraper-lite_2026-09-07_19-06-08-897.csv",
)
 
# ---------------------------------------------------------------------------
RE_BOILERPLATE = re.compile(r"submitted\s+by\s*/?u/\S+|\[link\]|\[comments\]", flags=re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_REMOVIDO = re.compile(r"^\s*\[(removed|deleted)\]\s*$", flags=re.IGNORECASE)
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
RE_USUARIO = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{3,}|(?<!\w)@[A-Za-z0-9_.]{3,}")
RE_ESPACOS = re.compile(r"\s+")
 
COLUNAS_SAIDA = [
    "id", "origem", "modo_coleta", "tipo", "communityName", "createdAt", "autor",
    "upVotes", "upVoteRatio", "numberOfComments",
    "titulo_limpo", "corpo_limpo", "texto", "n_palavras_corpo",
    "termos_busca", "contem_termo_busca",
]
 
log = []
 
 
def registrar(msg=""):
    print(msg)
    log.append(str(msg))
 
 
def carregar_sal():
    if os.path.exists(ARQUIVO_SAL):
        with open(ARQUIVO_SAL, encoding="utf-8") as f:
            return f.read().strip()
    sal = secrets.token_hex(16)
    os.makedirs(os.path.dirname(ARQUIVO_SAL), exist_ok=True)
    with open(ARQUIVO_SAL, "w", encoding="utf-8") as f:
        f.write(sal)
    registrar(f"Sal dos pseudônimos criado em {ARQUIVO_SAL} (não publique este arquivo).")
    return sal
 
 
def pseudonimo(nome, sal):
    """Troca o nome de usuário por um código estável e irreversível."""
    if pd.isna(nome) or not str(nome).strip():
        return ""
    return "user_" + hashlib.sha256((sal + str(nome)).encode()).hexdigest()[:10]
 
 
def limpar_texto(txt):
    txt = html.unescape(str(txt))            # &#32; -> espaço, &amp; -> &
    txt = RE_BOILERPLATE.sub(" ", txt)
    txt = RE_URL.sub(" ", txt)
    txt = RE_EMAIL.sub("[email]", txt)
    txt = RE_USUARIO.sub("[usuario]", txt)
    return RE_ESPACOS.sub(" ", txt).strip()
 
 
def sem_acento(txt):
    return unicodedata.normalize("NFD", str(txt).lower()).encode("ascii", "ignore").decode()
 
 
PADROES_BUSCA = {t: re.compile(r"(?<!\w)" + re.escape(sem_acento(t)) + r"(?!\w)") for t in dict.fromkeys(TERMOS_BUSCA)}
 
 
def termos_encontrados(txt):
    t = sem_acento(txt)
    return "; ".join(termo for termo, p in PADROES_BUSCA.items() if p.search(t))
 
 
def limpar_arquivo(df, nome_arquivo):
    n0 = len(df)
    if "dataType" in df.columns:
        df = df[df["dataType"].isin(TIPOS)]
 
    titulo = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    corpo = df.get("body", pd.Series("", index=df.index)).fillna("").astype(str)
    corpo_removido = corpo.str.match(RE_REMOVIDO)
 
    titulo = titulo.apply(limpar_texto)
    corpo = corpo.apply(limpar_texto)
    df = df.assign(
        tipo=df.get("dataType", "post"),
        titulo_limpo=titulo,
        corpo_limpo=corpo,
        texto=(titulo + ". " + corpo).str.strip(" ."),
        n_palavras_corpo=corpo.str.split().str.len(),
    )
    df = df[~corpo_removido]
    df = df[df["texto"].str.split().str.len() >= MIN_PALAVRAS]
    if EXIGIR_CORPO:
        df = df[df["n_palavras_corpo"] > 0]
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=["texto"])
 
    registrar(f"  {os.path.basename(nome_arquivo)}: {n0} -> {len(df)}")
    return df
 
 
def processar_pasta(pasta, origem):
    arquivos = sorted(glob.glob(f"{pasta}/*.csv"))
    registrar(f"\n--- {origem.upper()} ({len(arquivos)} arquivos em {pasta}) ---")
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV em {pasta}. Rode o script a partir da pasta do projeto.")
    partes, total_bruto = [], 0
    for f in arquivos:
        df_temp = pd.read_csv(f)
        total_bruto += len(df_temp)
        parte = limpar_arquivo(df_temp, f)
        if os.path.basename(f).endswith(EXTRACOES_POR_BUSCA):
            parte["modo_coleta"] = "busca_por_termo"
        else:
            parte["modo_coleta"] = "subreddits_gerais" if origem == "neutro" else "subreddits_selecionados"
        partes.append(parte)
    df = pd.concat(partes, ignore_index=True)
    df = df.drop_duplicates(subset=["id"]).drop_duplicates(subset=["texto"])   # entre arquivos
    df["origem"] = origem
    return df, total_bruto
 
 
if __name__ == "__main__":
    os.makedirs(DIR_SAIDA, exist_ok=True)
    sal = carregar_sal()
 
    df_neutro, bruto_neutro = processar_pasta(DIR_NEUTROS, "neutro")
    df_odio, bruto_odio = processar_pasta(DIR_ODIO, "odio")
 
    # um mesmo post nas duas coletas fica só na de ódio
    em_ambos = df_neutro["id"].isin(df_odio["id"]) | df_neutro["texto"].isin(df_odio["texto"])
    df_neutro = df_neutro[~em_ambos]
 
    df = pd.concat([df_neutro, df_odio], ignore_index=True)
    # o nome de usuário nunca sai deste script
    df["autor"] = df["username"].apply(lambda n: pseudonimo(n, sal)) if "username" in df.columns else ""
    df["termos_busca"] = df["texto"].map(termos_encontrados)
    df["contem_termo_busca"] = df["termos_busca"] != ""
    df = df[[c for c in COLUNAS_SAIDA if c in df.columns]]
    df.to_csv(f"{DIR_SAIDA}/reddit_limpo.csv", index=False, encoding="utf-8-sig")
 
    registrar("\n--- RESUMO ---")
    registrar(f"Coleta neutra : {bruto_neutro} brutos -> {len(df_neutro)} limpos "
              f"({int(em_ambos.sum())} removidos por também estarem na coleta de ódio)")
    registrar(f"Coleta de ódio: {bruto_odio} brutos -> {len(df_odio)} limpos")
    autores = df["autor"].fillna("").replace("", pd.NA)
    registrar(f"Total: {len(df)} textos | {df['communityName'].nunique()} subreddits | "
              f"{autores.nunique()} autores distintos ({autores.isna().mean():.0%} dos posts sem autor na coleta)")
    registrar("Posts que contêm algum termo de busca: "
              + " | ".join(f"{o}: {int(g.sum())} de {len(g)} ({g.mean():.0%})"
                           for o, g in df.groupby(["origem", "modo_coleta"])["contem_termo_busca"]))
    registrar(f"Mediana de palavras por texto: {df['texto'].str.split().str.len().median():.0f}")
    registrar(f"\nSaída: {DIR_SAIDA}/reddit_limpo.csv")
 
    with open(f"{DIR_SAIDA}/relatorio_limpeza.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))