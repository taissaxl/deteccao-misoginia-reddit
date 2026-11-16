import datetime
import json
import os
import re
import time

import pandas as pd

MODELO = os.environ.get("GEMINI_MODELO", "gemini-3.5-flash-lite")
NIVEL_RACIOCINIO = "low"
PAUSA = 7            # segundos entre chamadas
ARQ_AMOSTRA = "dados/processados/amostra_rotulagem.csv"
ARQ_PROMPT = "recursos/prompt_llm_anotacao.txt"
DIR = "resultados"
ARQ_BRUTAS = f"{DIR}/llm_respostas_brutas.jsonl"
ROTULOS_VALIDOS = {"misoginia", "odio_geral", "neutro", "pular"}

ESQUEMA = {
    "type": "object",
    "properties": {
        "rotulo": {"type": "string", "enum": sorted(ROTULOS_VALIDOS)},
        "tipo": {"type": ["string", "null"], "enum": ["explicita", "implicita", None]},
        "trecho": {"type": ["string", "null"]},
        "justificativa": {"type": "string"},
    },
    "required": ["rotulo", "tipo", "trecho", "justificativa"],
}

os.makedirs(DIR, exist_ok=True)


def extrair_json(texto):
    """Lê o JSON da resposta (tolera ```json ... ``` em volta). Devolve dict ou None."""
    m = re.search(r"\{.*\}", texto or "", flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if d.get("rotulo") not in ROTULOS_VALIDOS:
        return None
    if d["rotulo"] != "misoginia":
        d["tipo"], d["trecho"] = None, None
    elif d.get("tipo") not in ("explicita", "implicita"):
        return None
    return d


def nome(valor):
    """Enum do SDK -> texto ('STOP', 'SAFETY'...)."""
    if valor is None:
        return None
    return getattr(valor, "name", str(valor))


def configurar(types, prompt):
    sem_filtro = [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in (
        "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
    return types.GenerateContentConfig(
        system_instruction=prompt,
        thinking_config=types.ThinkingConfig(thinking_level=NIVEL_RACIOCINIO),
        response_mime_type="application/json",
        response_json_schema=ESQUEMA,
        safety_settings=sem_filtro,
    )


def chamar(cliente, config, texto, errors):
    """Faz a chamada, esperando e tentando de novo quando o limite por minuto estoura."""
    espera = 30
    for _ in range(6):
        try:
            return cliente.models.generate_content(
                model=MODELO, contents=f"Postagem:\n\n{texto}", config=config)
        except errors.APIError as e:
            if e.code == 429:
                if "PerDay" in str(e) or "per day" in str(e).lower():
                    raise SystemExit("Cota diária do nível gratuito esgotada. Rode de novo amanhã: "
                                     "o script continua de onde parou.")
                print(f"  limite por minuto atingido, esperando {espera}s")
                time.sleep(espera)
                espera = min(espera * 2, 120)
            elif e.code in (500, 503):
                time.sleep(espera)
            else:
                raise
    raise SystemExit("A API continuou recusando por limite de uso. Rode de novo mais tarde.")


def ler_resposta(resp):
    """Devolve (texto, motivo_fim, motivo_bloqueio)."""
    bloqueio = nome(getattr(getattr(resp, "prompt_feedback", None), "block_reason", None))
    cands = resp.candidates or []
    motivo = nome(cands[0].finish_reason) if cands else None
    try:
        texto = resp.text or ""
    except Exception:
        texto = ""
    return texto, motivo, bloqueio


def ja_respondidos():
    feitos = {}
    if os.path.exists(ARQ_BRUTAS):
        with open(ARQ_BRUTAS, encoding="utf-8") as f:
            for linha in f:
                r = json.loads(linha)
                if r.get("final"):
                    feitos[r["id"]] = r
    return feitos


if __name__ == "__main__":
    from google import genai
    from google.genai import errors, types

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Defina a variável de ambiente GEMINI_API_KEY com a sua chave da API.")
    cliente = genai.Client()
    prompt = open(ARQ_PROMPT, encoding="utf-8").read()
    config = configurar(types, prompt)
    amostra = pd.read_csv(ARQ_AMOSTRA)
    feitos = ja_respondidos()
    print(f"{len(amostra)} postagens | {len(feitos)} já respondidas | modelo {MODELO}")

    try:
        with open(ARQ_BRUTAS, "a", encoding="utf-8") as saida:
            for i, r in enumerate(amostra.itertuples(), 1):
                if r.id in feitos:
                    continue
                for tentativa in (1, 2):
                    resp = chamar(cliente, config, r.texto, errors)
                    time.sleep(PAUSA)
                    texto_resp, motivo, bloqueio = ler_resposta(resp)
                    parsed = extrair_json(texto_resp)
                    if bloqueio or motivo in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                        status = "bloqueio"
                    elif parsed is None:
                        status = "fora_formato"
                    else:
                        status = "ok"
                    final = status == "ok" or tentativa == 2
                    uso = resp.usage_metadata
                    registro = {
                        "id": r.id, "ordem": int(r.ordem), "tentativa": tentativa, "status": status,
                        "final": final, "modelo_resposta": getattr(resp, "model_version", None),
                        "motivo_fim": motivo, "motivo_bloqueio": bloqueio,
                        "texto_resposta": texto_resp, "resposta": parsed,
                        "tokens_entrada": getattr(uso, "prompt_token_count", None) or 0,
                        "tokens_saida": getattr(uso, "candidates_token_count", None) or 0,
                        "tokens_raciocinio": getattr(uso, "thoughts_token_count", None) or 0,
                        "data_hora": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                    saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    saida.flush()
                    if final:
                        feitos[r.id] = registro
                        break
                if i % 25 == 0:
                    print(f"  {i}/{len(amostra)}")
    except SystemExit as e:
        print(e)
        print(f"{len(feitos)} de {len(amostra)} postagens respondidas até agora.")
        raise

    # --- consolidação --------------------------------------------------------
    linhas = []
    for r in amostra.itertuples():
        reg = feitos[r.id]
        d = reg["resposta"] or {}
        linhas.append({"id": r.id, "ordem": int(r.ordem), "piloto": bool(r.piloto), "status": reg["status"],
                       "rotulo": d.get("rotulo"), "tipo": d.get("tipo"), "trecho": d.get("trecho"),
                       "justificativa": d.get("justificativa")})
    rot = pd.DataFrame(linhas)
    rot.to_csv(f"{DIR}/llm_rotulos.csv", index=False, encoding="utf-8-sig")

    todas = [json.loads(l) for l in open(ARQ_BRUTAS, encoding="utf-8")]
    tentativas = pd.DataFrame(todas)
    modelos = sorted(tentativas["modelo_resposta"].dropna().unique())
    datas = tentativas["data_hora"]
    t1 = tentativas["tentativa"] == 1
    rel = [
        f"Modelo solicitado: {MODELO} | versão informada pela API: {', '.join(modelos) or 'não informada'}",
        f"Execução: {datas.min()} a {datas.max()}",
        f"Configuração: thinking_level = {NIVEL_RACIOCINIO}, saída JSON com esquema, filtros de "
        "segurança desligados (BLOCK_NONE), temperatura padrão, zero-shot, uma chamada por postagem",
        f"Chamadas: {len(tentativas)} | tokens de entrada: {int(tentativas['tokens_entrada'].sum())} | "
        f"tokens de saída: {int(tentativas['tokens_saida'].sum())} | "
        f"tokens de raciocínio: {int(tentativas['tokens_raciocinio'].sum())}",
        "",
        f"Primeira tentativa: {t1.sum()} chamadas, "
        f"{(t1 & (tentativas['status'] == 'bloqueio')).sum()} bloqueios, "
        f"{(t1 & (tentativas['status'] == 'fora_formato')).sum()} fora do formato",
        f"Resultado final: {rot['status'].value_counts().to_dict()}",
        f"Rótulos: {rot['rotulo'].value_counts(dropna=False).to_dict()}",
    ]
    open(f"{DIR}/relatorio_llm.txt", "w", encoding="utf-8").write("\n".join(rel))
    print("\n".join(rel))