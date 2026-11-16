import datetime
import importlib
import json
import os
import time
 
import pandas as pd
 
base = importlib.import_module("08_llm_anotacao")   # mesma configuração do 08
 
ARQ_REDDIT = "dados/processados/reddit_limpo.csv"
ARQ_BRUTAS = f"{base.DIR}/llm_reddit_respostas_brutas.jsonl"
 
 
def ler_finais(caminho):
    feitos = {}
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
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
    reddit = pd.read_csv(ARQ_REDDIT)
    feitos = ler_finais(ARQ_BRUTAS)
 
    # reaproveita as respostas da amostra (08) que ainda não estão neste arquivo
    amostra = ler_finais(base.ARQ_BRUTAS)
    novas = {i: r for i, r in amostra.items() if i in set(reddit["id"]) and i not in feitos}
    with open(ARQ_BRUTAS, "a", encoding="utf-8") as saida:
        for i, r in novas.items():
            r = {**r, "reaproveitada_da_amostra": True}
            saida.write(json.dumps(r, ensure_ascii=False) + "\n")
            feitos[i] = r
    print(f"{len(reddit)} postagens | {len(feitos)} já respondidas ({len(novas)} reaproveitadas agora "
          f"da amostra) | modelo {base.MODELO}")
 
    cliente = genai.Client()
    config = base.configurar(types, open(base.ARQ_PROMPT, encoding="utf-8").read())
    try:
        with open(ARQ_BRUTAS, "a", encoding="utf-8") as saida:
            for n, r in enumerate(reddit.itertuples(), 1):
                if r.id in feitos:
                    continue
                for tentativa in (1, 2):
                    resp = base.chamar(cliente, config, r.texto, errors)
                    time.sleep(base.PAUSA)
                    texto_resp, motivo, bloqueio = base.ler_resposta(resp)
                    parsed = base.extrair_json(texto_resp)
                    if bloqueio or motivo in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
                        status = "bloqueio"
                    elif parsed is None:
                        status = "fora_formato"
                    else:
                        status = "ok"
                    final = status == "ok" or tentativa == 2
                    uso = resp.usage_metadata
                    registro = {
                        "id": r.id, "tentativa": tentativa, "status": status, "final": final,
                        "modelo_resposta": getattr(resp, "model_version", None),
                        "motivo_fim": motivo, "motivo_bloqueio": bloqueio,
                        "texto_resposta": texto_resp, "resposta": parsed,
                        "tokens_entrada": getattr(uso, "prompt_token_count", None) or 0,
                        "tokens_saida": getattr(uso, "candidates_token_count", None) or 0,
                        "tokens_raciocinio": getattr(uso, "thoughts_token_count", None) or 0,
                        "data_hora": datetime.datetime.now().isoformat(timespec="seconds"),
                        "reaproveitada_da_amostra": False,
                    }
                    saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    saida.flush()
                    if final:
                        feitos[r.id] = registro
                        break
                if n % 50 == 0:
                    print(f"  {n}/{len(reddit)} ({len(feitos)} respondidas)")
    except SystemExit as e:
        print(e)
        print(f"{len(feitos)} de {len(reddit)} postagens respondidas até agora.")
        raise
 
    # --- consolidação --------------------------------------------------------
    linhas = []
    for r in reddit.itertuples():
        reg = feitos[r.id]
        d = reg["resposta"] or {}
        linhas.append({"id": r.id, "status": reg["status"], "rotulo": d.get("rotulo"), "tipo": d.get("tipo"),
                       "trecho": d.get("trecho"), "justificativa": d.get("justificativa"),
                       "reaproveitada_da_amostra": bool(reg.get("reaproveitada_da_amostra"))})
    rot = pd.DataFrame(linhas)
    rot.to_csv(f"{base.DIR}/llm_rotulos_reddit.csv", index=False, encoding="utf-8-sig")
 
    t = pd.DataFrame([json.loads(l) for l in open(ARQ_BRUTAS, encoding="utf-8")])
    novas_ch = t[t["reaproveitada_da_amostra"] != True]  # noqa: E712
    modelos = sorted(t["modelo_resposta"].dropna().unique())
    rel = [
        f"Modelo solicitado: {base.MODELO} | versão informada pela API: {', '.join(modelos) or 'não informada'}",
        f"Postagens: {len(rot)} | reaproveitadas da amostra (08): {int(rot['reaproveitada_da_amostra'].sum())} | "
        f"enviadas nesta etapa: {len(rot) - int(rot['reaproveitada_da_amostra'].sum())}",
        f"Execução desta etapa: {novas_ch['data_hora'].min()} a {novas_ch['data_hora'].max()}",
        f"Chamadas desta etapa: {len(novas_ch)} | bloqueios na 1ª tentativa: "
        f"{((novas_ch['tentativa'] == 1) & (novas_ch['status'] == 'bloqueio')).sum()} | fora do formato na 1ª: "
        f"{((novas_ch['tentativa'] == 1) & (novas_ch['status'] == 'fora_formato')).sum()}",
        f"Resultado final: {rot['status'].value_counts().to_dict()}",
        f"Rótulos: {rot['rotulo'].value_counts(dropna=False).to_dict()}",
    ]
    open(f"{base.DIR}/relatorio_llm_reddit.txt", "w", encoding="utf-8").write("\n".join(rel))
    print("\n".join(rel))