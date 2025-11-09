# -*- coding: utf-8 -*-
import os
import math
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ===== CONFIGURAÇÕES =====
CSV_PATH = "./dados/crowdsourced_pt.csv"      # caminho do CSV
TEXT_COL = "text_pt"           # coluna com o texto a classificar
LABEL_COL = "Verdict"          # coluna com o rótulo de verdade-terreno
MODEL_ID = "Sami92/XLM-R-Large-ClaimDetection"
BATCH_SIZE = 32                # ajuste conforme memória
MAX_LEN = 512                  # limite do XLM-R
DEVICE = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") else -1  # GPU se houver

# ===== MAPA DE RÓTULOS =====
def normalize_gold_label(x):
    """
    Converte rótulos diversos para {'factual','non-factual'}.
    Casos comuns:
      1 -> factual
      0 -> non-factual
     -1 -> non-factual
     'factual'/'non-factual' (ou variações de caixa) -> mantém
    """
    if pd.isna(x):
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"factual", "fact", "true"}:
            return "factual"
        if s in {"non-factual", "nonfactual", "not-factual", "false", "fake"}:
            return "non-factual"
        # tentativa de converter strings numéricas
        try:
            v = int(float(s))
            return "factual" if v == 1 else "non-factual"
        except:
            return None
    # numérico
    try:
        v = int(x)
        return "factual" if v == 1 else "non-factual"
    except:
        return None

# ===== CARREGAR DADOS =====
df = pd.read_csv(CSV_PATH)
if TEXT_COL not in df.columns:
    raise ValueError(f"Coluna '{TEXT_COL}' não encontrada no CSV.")
if LABEL_COL not in df.columns:
    raise ValueError(f"Coluna '{LABEL_COL}' não encontrada no CSV.")

df = df[[TEXT_COL, LABEL_COL]].copy()
df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()
df["gold"] = df[LABEL_COL].apply(normalize_gold_label)
df = df.dropna(subset=[TEXT_COL, "gold"]).reset_index(drop=True)

texts = df[TEXT_COL].tolist()
golds = df["gold"].tolist()

# ===== PIPELINE DO MODELO =====
tokenizer_kwargs = dict(truncation=True, max_length=MAX_LEN)
clf = pipeline(
    "text-classification",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=DEVICE,
    **tokenizer_kwargs,
)

# ===== INFERÊNCIA EM LOTE =====
preds = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Inferindo"):
    batch = texts[i:i+BATCH_SIZE]
    out = clf(batch)
    # out é uma lista de dicts [{'label': 'factual'|'non-factual', 'score': float}, ...]
    preds.extend([o["label"].strip().lower() for o in out])

# ===== RELATÓRIO =====
print(classification_report(golds, preds, digits=2))
print("Matriz de confusão (linhas=verdade, colunas=previsto):")
print(confusion_matrix(golds, preds, labels=["factual","non-factual"]))
