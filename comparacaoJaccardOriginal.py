# coding: utf-8
# ================================================================
# Cálculo do Jaccard entre:
#   - kw_orig: keywords por texto vindas de docs_rafael.xlsx (coluna "Palavras-chaves")
#   - kw_res : keywords extraídas dos RESUMOS gerados pelos modelos
#
# Saídas:
#   - resultados2/jaccard_por_texto.xlsx
#   - resultados2/jaccard_por_modelo.xlsx
# ================================================================

import os
import re
import pandas as pd
import numpy as np

# --- sumarização (para gerar os resumos) ---
from summarizer.sbert import SBertSummarizer
from summarizer import Summarizer

# --- extração de keywords dos resumos ---
from keybert import KeyBERT


# ====================== Funções utilitárias ======================
def _split_keywords(cell):
    """Converte a célula do Excel (string, lista, etc.) em um conjunto de keywords (minúsculas)."""
    if pd.isna(cell):
        return set()
    if isinstance(cell, (list, tuple, set)):
        return {str(x).strip().lower() for x in cell}
    # separa por vírgula, ponto-e-vírgula, barras, pipe ou quebras de linha
    toks = re.split(r'[;,|/\n]+', str(cell))
    return {t.strip().lower() for t in toks if t.strip()}

def kw_from_excel(docs: pd.DataFrame, i: int, col: str = "Palavras-chaves") -> set:
    """
    Retorna o conjunto de keywords de docs_rafael.xlsx para o texto i.
    Se existir a coluna 'idx_texto', usa como chave; caso contrário, usa a linha i.
    """
    try:
        if "idx_texto" in docs.columns:
            return _split_keywords(docs.set_index("idx_texto").at[i, col])
        return _split_keywords(docs.at[i, col])
    except Exception:
        return set()

def jaccard(a: set, b: set) -> float:
    inter = len(a & b)
    uni = len(a | b) or 1
    return inter / uni

def resumo_keywords(texto: str, kb: KeyBERT, topn: int = 10) -> set:
    """
    Extrai keywords (1 a 3-gramas) do texto usando KeyBERT e retorna como conjunto minúsculo.
    """
    pairs = kb.extract_keywords(texto, top_n=topn, keyphrase_ngram_range=(1, 3))
    # pairs pode ser [('palavra', score), ...] ou lista simples dependendo da versão
    kws = []
    for p in pairs:
        if isinstance(p, (list, tuple)) and len(p) >= 1:
            kws.append(str(p[0]).lower())
        else:
            kws.append(str(p).lower())
    return set(kws)


# ====================== Entradas ======================
# textos.txt: uma linha por texto
# docs_rafael.xlsx: deve ter a coluna "Palavras-chaves" e (opcional) "idx_texto"
def carregar_textos(caminho: str) -> list:
    # tenta latin1 primeiro (igual ao seu script), senão usa utf-8
    for enc in ("latin1", "utf-8"):
        try:
            with open(caminho, "r", encoding=enc) as f:
                return [linha.strip() for linha in f.readlines() if linha.strip()]
        except Exception:
            continue
    raise RuntimeError(f"Não foi possível ler {caminho} com latin1 nem utf-8.")

noticias = carregar_textos("textos.txt")
docs_rafael = pd.read_excel("docs_rafael.xlsx")  # coluna esperada: "Palavras-chaves"

# ====================== Modelos (somente para gerar RESUMOS) ======================
modelos = {
    "SBERT-MiniLM": SBertSummarizer('paraphrase-MiniLM-L6-v2'),
    "DistilBERT":  Summarizer('distilbert-base-uncased', hidden=[-1, -2], hidden_concat=True),
    "BERT-base":   Summarizer()
}

kw_model = KeyBERT()

# ====================== Loop principal: calcular apenas Jaccard ======================
registros = []

os.makedirs("resultados2", exist_ok=True)

for nome_modelo, mdl in modelos.items():
    print(f">>> Rodando Jaccard para modelo: {nome_modelo}")
    for i, original in enumerate(noticias):
        try:
            # 1) gerar resumo
            resumo_m = mdl(original)

            # 2) keywords do resumo (kw_res) e do Excel (kw_orig)
            kw_res  = resumo_keywords(resumo_m, kw_model, topn=10)
            kw_orig = kw_from_excel(docs_rafael, i, col="Palavras-chaves")

            # 3) Jaccard
            jacc = jaccard(kw_orig, kw_res)

            registros.append({
                "modelo": nome_modelo,
                "idx_texto": i,
                "jaccard_kw": jacc,
                "kw_orig": "; ".join(sorted(kw_orig)) if kw_orig else "",
                "kw_resumo": "; ".join(sorted(kw_res)) if kw_res else ""
            })
        except Exception as e:
            print(f"[ERRO] Modelo={nome_modelo} texto#{i}: {e}")

# ====================== Saídas: somente Jaccard ======================
df_jaccard = pd.DataFrame(registros)

if df_jaccard.empty:
    raise SystemExit("Sem resultados de Jaccard. Verifique entradas/modelos.")

df_jaccard.to_excel("resultados2/jaccard_por_texto.xlsx", index=False)

agg = (df_jaccard
       .groupby("modelo", as_index=False)
       .agg(jaccard_kw_medio=("jaccard_kw", "mean")))

agg.to_excel("resultados2/jaccard_por_modelo.xlsx", index=False)

print("\nArquivos gerados em /jaccardResults:")
print("- jaccard_por_texto.xlsx")
print("- jaccard_por_modelo.xlsx")
