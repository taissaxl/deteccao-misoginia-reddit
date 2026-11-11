import inspect
import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                          TrainingArguments)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
MODELO_BASE = "neuralmind/bert-base-portuguese-cased"
DIR_BASES = "dados/bases/processadas"
DIR_MODELO = "modelos/bertimbau-misoginia"
DIR_RESULTADOS = "resultados"

SEED = 42
MAX_LEN = 256          # tokens por texto no treino
TAXA_APRENDIZADO = 2e-5
LOTE = 16
EPOCAS_MAX = 6
PACIENCIA = 3

os.makedirs(DIR_MODELO, exist_ok=True)
os.makedirs(DIR_RESULTADOS, exist_ok=True)


def fixar_semente(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


fixar_semente(SEED)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {dispositivo}" + (f" ({torch.cuda.get_device_name(0)})" if dispositivo == "cuda" else ""))
if dispositivo == "cpu":
    print("ATENÇÃO: sem GPU o treino leva horas. No Colab: Ambiente de execução > Alterar o tipo > GPU T4.")

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
treino = pd.read_csv(f"{DIR_BASES}/treino.csv")
validacao = pd.read_csv(f"{DIR_BASES}/validacao.csv")
teste = pd.read_csv(f"{DIR_BASES}/teste.csv")
for nome, df in [("treino", treino), ("validação", validacao), ("teste", teste)]:
    print(f"{nome:10s}: {len(df):6d} textos | misoginia = {df['label'].sum():5d} ({df['label'].mean():.1%})")

tokenizador = AutoTokenizer.from_pretrained(MODELO_BASE)


class Textos(torch.utils.data.Dataset):
    def __init__(self, df):
        self.enc = tokenizador(df["texto"].astype(str).tolist(), truncation=True, max_length=MAX_LEN)
        self.labels = df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


ds_treino, ds_val, ds_teste = Textos(treino), Textos(validacao), Textos(teste)

# pesos das classes: inversamente proporcionais à frequência no treino
contagem = np.bincount(treino["label"], minlength=2)
pesos = torch.tensor(len(treino) / (2 * contagem), dtype=torch.float)
print(f"Pesos das classes (0, 1): {pesos.tolist()}")

# ---------------------------------------------------------------------------
# Modelo e treino
# ---------------------------------------------------------------------------
modelo = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BASE, num_labels=2,
    id2label={0: "nao_misoginia", 1: "misoginia"}, label2id={"nao_misoginia": 0, "misoginia": 1})


class TrainerComPesos(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        saidas = model(**inputs)
        perda = nn.CrossEntropyLoss(weight=pesos.to(saidas.logits.device))(saidas.logits, labels)
        return (perda, saidas) if return_outputs else perda


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)   # com fp16 os logits vêm em meia precisão
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def metricas(pred):
    logits, y = pred
    p = softmax(logits)[:, 1]
    yhat = (p >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return {"f1_misoginia": f1, "precisao_misoginia": prec, "revocacao_misoginia": rec,
            "f1_macro": f1_score(y, yhat, average="macro"), "auc": roc_auc_score(y, p)}


# compatibilidade entre versões do transformers
params_args = inspect.signature(TrainingArguments.__init__).parameters
chave_eval = "eval_strategy" if "eval_strategy" in params_args else "evaluation_strategy"
# compatibilidade entre versões do transformers: o aquecimento (warmup) de 10% dos passos
# é passado como warmup_ratio nas versões antigas e em passos (warmup_steps) nas novas
passos_totais = math.ceil(len(ds_treino) / LOTE) * EPOCAS_MAX
config_args = dict(
    output_dir="checkpoints",
    learning_rate=TAXA_APRENDIZADO,
    per_device_train_batch_size=LOTE,
    per_device_eval_batch_size=LOTE * 4,
    num_train_epochs=EPOCAS_MAX,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_misoginia",
    greater_is_better=True,
    logging_steps=100,
    fp16=(dispositivo == "cuda"),
    seed=SEED,
    report_to="none",
    **{chave_eval: "epoch"},
)
if "warmup_ratio" in params_args:
    config_args["warmup_ratio"] = 0.1
else:
    config_args["warmup_steps"] = int(0.1 * passos_totais)
ignorados = [k for k in config_args if k not in params_args]
if ignorados:
    print(f"Aviso: parâmetros não aceitos por esta versão do transformers e ignorados: {ignorados}")
args = TrainingArguments(**{k: v for k, v in config_args.items() if k in params_args})
params_trainer = inspect.signature(Trainer.__init__).parameters
chave_tok = "processing_class" if "processing_class" in params_trainer else "tokenizer"
trainer = TrainerComPesos(
    model=modelo,
    args=args,
    train_dataset=ds_treino,
    eval_dataset=ds_val,
    data_collator=DataCollatorWithPadding(tokenizador),
    compute_metrics=metricas,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PACIENCIA)],
    **{chave_tok: tokenizador},
)

trainer.train()
trainer.save_model(DIR_MODELO)
tokenizador.save_pretrained(DIR_MODELO)
hist = pd.DataFrame([h for h in trainer.state.log_history if "eval_f1_misoginia" in h])
hist.to_csv(f"{DIR_RESULTADOS}/historico_treino.csv", index=False)
print("\nHistórico por época:")
print(hist[[c for c in ["epoch", "eval_loss", "eval_f1_misoginia", "eval_precisao_misoginia",
                        "eval_revocacao_misoginia", "eval_auc"] if c in hist]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# Limiar e avaliação no teste
# ---------------------------------------------------------------------------
p_val = softmax(trainer.predict(ds_val).predictions)[:, 1]
limiares = np.arange(0.05, 0.96, 0.01)
f1s = [f1_score(validacao["label"], (p_val >= t).astype(int), zero_division=0) for t in limiares]
limiar = float(limiares[int(np.argmax(f1s))])
print(f"\nLimiar que maximiza o F1 na validação: {limiar:.2f} (F1 = {max(f1s):.3f})")

p_teste = softmax(trainer.predict(ds_teste).predictions)[:, 1]
y = teste["label"].values
resultado = {"modelo_base": MODELO_BASE, "max_len": MAX_LEN, "taxa_aprendizado": TAXA_APRENDIZADO,
             "lote": LOTE, "epocas_treinadas": float(trainer.state.epoch), "semente": SEED,
             "limiar_validacao": limiar, "auc": float(roc_auc_score(y, p_teste))}
texto_relatorio = []
for nome, t in [("limiar_0.5", 0.5), ("limiar_validacao", limiar)]:
    yhat = (p_teste >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    resultado[nome] = {"limiar": t, "f1_misoginia": f1, "precisao_misoginia": prec,
                       "revocacao_misoginia": rec, "f1_macro": f1_score(y, yhat, average="macro")}
    texto_relatorio += [f"=== Teste (TuPy-E oficial) | limiar {t:.2f} ===",
                        classification_report(y, yhat, target_names=["nao_misoginia", "misoginia"], digits=3),
                        "Matriz de confusão (linhas = real, colunas = previsto):",
                        str(confusion_matrix(y, yhat)), ""]
texto_relatorio.append(f"AUC: {resultado['auc']:.3f}")
print("\n" + "\n".join(texto_relatorio))

with open(f"{DIR_RESULTADOS}/metricas_teste.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2, default=float)
with open(f"{DIR_RESULTADOS}/relatorio_teste.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(texto_relatorio))
teste.assign(prob_misoginia=p_teste, previsto=(p_teste >= limiar).astype(int)).to_csv(
    f"{DIR_RESULTADOS}/predicoes_teste.csv", index=False, encoding="utf-8")
# guarda o limiar junto com o modelo, para o script de avaliação do Reddit
with open(f"{DIR_MODELO}/limiar.json", "w", encoding="utf-8") as f:
    json.dump({"limiar": limiar}, f)


print(f"\nModelo salvo em {DIR_MODELO}/ e resultados em {DIR_RESULTADOS}/")