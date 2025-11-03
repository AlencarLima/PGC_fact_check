# coding: utf-8
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from collections import Counter

# Modelos de sumarização
from summarizer.sbert import SBertSummarizer
from summarizer import Summarizer

# Palavras-chave / NLP
from keybert import KeyBERT
import nltk
from nltk.corpus import mac_morpho, stopwords
from nltk.tag import UnigramTagger, BigramTagger
from nltk import word_tokenize

# Métricas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Classificação de “factual”
from transformers import pipeline

def carregar_resumos_da_coluna_f(caminho_xlsx):
    df = pd.read_excel(caminho_xlsx)      # 1ª linha é o header → ok
    col_f = df.iloc[:, 5]                 # coluna F = índice 5 (0=A, 1=B, ..., 5=F)

    resumos = []
    for cel in col_f:
        if pd.isna(cel):
            resumos.append("")            # célula vazia → resumo vazio
            continue

        # caso mais comum: a célula é uma string que PARECE uma lista
        if isinstance(cel, str):
            try:
                possivel_lista = ast.literal_eval(cel)
                if isinstance(possivel_lista, (list, tuple)):
                    texto = " ".join(str(x) for x in possivel_lista)
                else:
                    # era string mas não era lista → usa como está
                    texto = str(possivel_lista)
            except (SyntaxError, ValueError):
                # não era uma lista em string → usa o próprio texto
                texto = cel
        # se por acaso já vier como lista mesmo
        elif isinstance(cel, (list, tuple)):
            texto = " ".join(str(x) for x in cel)
        else:
            texto = str(cel)

        resumos.append(texto)

    return resumos

# ====================== 0) Utilidades e Setup ======================
def ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/mac_morpho')
    except LookupError:
        nltk.download('mac_morpho')

ensure_nltk()

def treinar_tagger_portugues():
    treino = mac_morpho.tagged_sents()
    unigram_tagger = UnigramTagger(treino)
    bigram_tagger = BigramTagger(treino, backoff=unigram_tagger)
    return bigram_tagger

def jaccard(a:set, b:set):
    inter = len(a & b)
    uni = len(a | b) or 1
    return inter / uni

def extrair_palavras_chaves(texto, tagger, keyBert):
    tokens = word_tokenize(texto)
    pos_tags = tagger.tag(tokens)
    if pos_tags is None:
        return [], [], [], []

    stop_words = list(stopwords.words('portuguese'))
    frases_nominais, frase_atual = [], []

    for palavra, tag in pos_tags:
        if tag in ('N', 'NPROP', 'ADJ') and palavra.lower() not in stop_words:
            frase_atual.append(palavra)
        else:
            if frase_atual:
                frases_nominais.append(' '.join(frase_atual))
                frase_atual = []
    if frase_atual:
        frases_nominais.append(' '.join(frase_atual))

    contagem = Counter(tok.lower() for tok in tokens if tok.lower() not in stop_words and tok.isalpha())
    principais_palavras = contagem.most_common(10)

    texto_unido = ' '.join(frases_nominais) if frases_nominais else texto
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform([texto_unido])
    tfidf_scores = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out(), columns=["score"])
    principais_tfidf = tfidf_scores.nlargest(10, "score").index.tolist()

    palavras_chaves_bert = keyBert.extract_keywords(texto)
    return frases_nominais, principais_palavras, principais_tfidf, palavras_chaves_bert

def resumo_to_keywords(texto_resumo, tagger, keyBert, topn=10):
    _, _, _, kw_bert = extrair_palavras_chaves(texto_resumo, tagger, keyBert)
    kw = [k[0] if isinstance(k, (list, tuple)) else str(k) for k in kw_bert]
    return set([k.lower() for k in kw[:topn]])

def similaridade_tfidf(original, resumo):
    vect = TfidfVectorizer(stop_words=list(stopwords.words('portuguese')))
    X = vect.fit_transform([original, resumo])
    return cosine_similarity(X[0], X[1])[0, 0]


# Quebra por conjunções/pontuação para claim detection
conjuncao_pattern = (
    r'\b(e|nem|também|bem como|não só\.\.\.mas|também|mas|porém|contudo|todavia|entretanto|'
    r'no entanto|não obstante|ou|ou\.\.\.ou|já\.\.\.já|ora\.\.\.ora|quer\.\.\.quer|seja\.\.\.seja|'
    r'logo|pois|portanto|assim|por isso|por consequência|por conseguinte|porque|que|porquanto|'
    r'visto que|uma vez que|já que|pois que|como|tanto que|tão que|tal que|tamanho que|de forma que|'
    r'de modo que|de sorte que|de tal forma que|a fim de que|para que|quando|enquanto|agora que|'
    r'logo que|desde que|assim que|apenas|se|caso|desde|salvo se|exceto se|contanto que|embora|'
    r'conquanto|ainda que|mesmo que|se bem que|posto que|assim como|tal|qual|tanto como|conforme|'
    r'consoante|segundo|à proporção que|à medida que|ao passo que|quanto mais\.\.\.mais)\b|[\.!?]'
)


# ====================== 1) Definir modelos a comparar ======================
modelos = {
    "SBERT-MiniLM": SBertSummarizer('paraphrase-MiniLM-L6-v2'),
    "DistilBERT": Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True),
    "BERT-base": Summarizer()
}


# ====================== 2) Inicializações fora do loop ======================
kw_model = KeyBERT()
tagger = treinar_tagger_portugues()

# claim detection 1x
checkpoint = "Sami92/XLM-R-Large-ClaimDetection"
tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
try:
    claimdetection = pipeline(
        "text-classification",
        model=checkpoint,
        tokenizer=checkpoint,
        **tokenizer_kwargs,
        device=-1  # CPU; mude para 0 se tiver GPU
    )
except Exception as e:
    print(f"[AVISO] Falha ao carregar pipeline de claim detection: {e}")
    claimdetection = None  # segue sem factual_share

def factual_share(texto):
    if not claimdetection:
        return np.nan
    partes = re.split(conjuncao_pattern, texto)
    partes = [p.strip() for p in partes if p and p.strip()]
    if not partes:
        return 0.0
    try:
        results = claimdetection(partes)
        factual = sum(1 for r in results if str(r.get('label', '')).lower() == 'factual')
        return factual / len(results)
    except Exception:
        return np.nan


# ====================== 3) Rodar modelos e coletar métricas ======================
with open('textos.txt', 'r', encoding='latin1') as arquivo:
    noticias = [linha.strip() for linha in arquivo.readlines() if linha.strip()]

os.makedirs("resultados2/por_texto", exist_ok=True)

registros = []
resumos_reg = []  # para comparativos de tamanho por texto

# Pré-calcula baseline factual dos originais (para não chamar pipeline à toa)
factual_orig_por_texto = [factual_share(orig) for orig in noticias]

for nome_modelo, mdl in modelos.items():
    print(f"\n>>> Rodando modelo: {nome_modelo}")
    for i, original in enumerate(noticias):
        try:
            t0 = time.time()
            resumo_m = mdl(original)
            dur = time.time() - t0

            comp = (len(resumo_m) / max(len(original), 1))
            sim = similaridade_tfidf(original, resumo_m)

            kw_orig = resumo_to_keywords(original, tagger, kw_model, topn=10)
            kw_res  = resumo_to_keywords(resumo_m, tagger, kw_model, topn=10)
            jacc = jaccard(kw_orig, kw_res)

            fact = factual_share(resumo_m)

            registros.append({
                "modelo": nome_modelo,
                "idx_texto": i,
                "tempo_s": dur,
                "compression_ratio": comp,
                "sim_tfidf": sim,
                "jaccard_kw": jacc,
                "factual_share": fact
            })

            # Guarda info para comparativo de tamanho por texto
            resumos_reg.append({
                "idx_texto": i,
                "modelo": nome_modelo,
                "len_texto": len(resumo_m),
                "tipo": "Resumo"
            })

        except Exception as e:
            print(f"[ERRO] Modelo={nome_modelo} texto#{i}: {e}")

resumos_excel = carregar_resumos_da_coluna_f("queTextos.xlsx")

for i, original in enumerate(noticias):
    try:
        # pega o resumo da linha i da coluna F
        resumo_m = resumos_excel[i]

        t0 = time.time()
        # aqui NÃO chamamos mais o modelo, porque o resumo já veio do Excel
        dur = time.time() - t0   # vai dar bem pequeno, mas mantém o campo

        comp = (len(resumo_m) / max(len(original), 1))
        sim  = similaridade_tfidf(original, resumo_m)

        kw_orig = resumo_to_keywords(original, tagger, kw_model, topn=10)
        kw_res  = resumo_to_keywords(resumo_m, tagger, kw_model, topn=10)
        jacc    = jaccard(kw_orig, kw_res)

        fact = factual_share(resumo_m)

        registros.append({
            "modelo": "'que'",
            "idx_texto": i,
            "tempo_s": dur,
            "compression_ratio": comp,
            "sim_tfidf": sim,
            "jaccard_kw": jacc,
            "factual_share": fact
        })

        resumos_reg.append({
            "idx_texto": i,
            "modelo": nome_modelo,
            "len_texto": len(resumo_m),
            "tipo": "Resumo"
        })

    except IndexError:
        # caso o Excel tenha menos linhas que 'noticias'
        break

# ---------- Baseline "Original" (comparação direta com os textos originais) ----------
# Adiciona uma linha por texto para o pseudo-modelo "Original"
for i, original in enumerate(noticias):
    registros.append({
        "modelo": "Original",
        "idx_texto": i,
        "tempo_s": 0.0,                       # não há custo de sumarização
        "compression_ratio": 1.0,             # |original| / |original|
        "sim_tfidf": 1.0,                     # similaridade do original com ele mesmo
        "jaccard_kw": 1.0,                    # KW(original) vs KW(original)
        "factual_share": factual_orig_por_texto[i]
    })
    resumos_reg.append({
        "idx_texto": i,
        "modelo": "Original",
        "len_texto": len(original),
        "tipo": "Original"
    })


df_metrics = pd.DataFrame(registros)
df_lens    = pd.DataFrame(resumos_reg)

if df_metrics.empty:
    raise SystemExit("Sem métricas (df_metrics vazio). Verifique entradas/modelos.")

os.makedirs("resultados2", exist_ok=True)
df_metrics.to_excel("resultados2/metrics_por_texto_com_baseline.xlsx", index=False)

# Agregado por modelo (agora incluindo 'Original')
agg = (df_metrics
       .groupby("modelo", as_index=False)
       .agg(tempo_medio_s=("tempo_s","mean"),
            compression_ratio_medio=("compression_ratio","mean"),
            sim_tfidf_medio=("sim_tfidf","mean"),
            jaccard_kw_medio=("jaccard_kw","mean"),
            factual_share_medio=("factual_share","mean"))
      )
agg.to_excel("resultados2/metrics_por_modelo_com_baseline.xlsx", index=False)
print("\nResumo por modelo (inclui baseline 'Original'):")
print(agg)


# ====================== 4) Gráficos (barras + radar) ======================
def plot_bar(df, col, titulo, ylabel, fname):
    xs = np.arange(len(df))
    vals = df[col].values
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(xs, vals)
    ax.bar_label(bars, fmt='{:,.2f}', padding=3, color='black', fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(df["modelo"].values, rotation=15)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ymax = max(vals) if len(vals) else 1
    ax.set_ylim(0, ymax*1.1 if ymax>0 else 1)
    plt.tight_layout()
    plt.savefig(f"resultados2/{fname}.png", dpi=160)
    plt.close()

# Barras agregadas (AGORA COM 'Original')
plot_bar(agg, "tempo_medio_s", "Tempo médio por texto (inclui Original)", "segundos", "tempo_medio_com_original")
plot_bar(agg, "sim_tfidf_medio", "Similaridade TF-IDF média (inclui Original)", "cosseno", "sim_tfidf_medio_com_original")
plot_bar(agg, "jaccard_kw_medio", "Overlap de palavras-chave (Jaccard) — (inclui Original)", "índice (0–1)", "jaccard_kw_medio_com_original")
plot_bar(agg, "factual_share_medio", "% trechos factuais (inclui Original)", "proporção (0–1)", "factual_share_medio_com_original")
plot_bar(agg, "compression_ratio_medio", "Compression ratio médio (inclui Original)", "|resumo|/|original|", "compression_ratio_medio_com_original")


# Radar — geramos DUAS versões: sem baseline e com baseline
def minmax_norm(x):
    mn, mx = np.nanmin(x), np.nanmax(x)
    return (x - mn) / (mx - mn + 1e-9)

def radar_plot(df_in, titulo, fname):
    radar_cols = ["sim_tfidf_medio", "jaccard_kw_medio", "factual_share_medio", "compression_ratio_medio", "tempo_medio_s"]
    norm = df_in.copy()
    for c in radar_cols:
        if c != "tempo_medio_s":
            norm[c] = minmax_norm(norm[c].values)
        else:
            norm[c] = 1 - minmax_norm(norm[c].values)  # menor tempo = melhor
    labels = ["Sim TF-IDF", "Jaccard KW", "% Factual", "Compressão", "Velocidade"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    plt.figure(figsize=(7,7))
    ax = plt.subplot(111, polar=True)
    for _, row in norm.iterrows():
        vals = row[radar_cols].values
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    plt.title(titulo)
    plt.legend(norm["modelo"].values, loc="upper right", bbox_to_anchor=(1.25,1.1))
    plt.tight_layout()
    plt.savefig(f"resultados2/{fname}.png", dpi=160)
    plt.close()

# Radar sem baseline "Original" (apenas modelos)
radar_plot(agg[agg["modelo"]!="Original"], "Radar de desempenho (modelos)", "radar_modelos")

# Radar com baseline "Original"
radar_plot(agg, "Radar de desempenho (modelos + Original)", "radar_com_original")


# ====================== 5) Boxplots por métrica (inclui Original) ======================
def boxplot_metric(df, col, ylabel, fname):
    modelos_ord = df['modelo'].unique()
    data = [df.loc[df['modelo']==m, col].dropna().values for m in modelos_ord]
    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=modelos_ord, showmeans=True)
    plt.title(f"Distribuição por texto — {ylabel} (inclui Original)")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"resultados2/box_{fname}.png", dpi=160)
    plt.close()

boxplot_metric(df_metrics, 'sim_tfidf', 'Similaridade TF-IDF', 'sim_tfidf_com_original')
boxplot_metric(df_metrics, 'jaccard_kw', 'Jaccard KW', 'jaccard_kw_com_original')
boxplot_metric(df_metrics, 'compression_ratio', 'Compression ratio', 'compression_ratio_com_original')
boxplot_metric(df_metrics, 'tempo_s', 'Tempo (s)', 'tempo_s_com_original')
boxplot_metric(df_metrics, 'factual_share', '% Factual', 'factual_share_com_original')


# ====================== 6) Scatter trade-offs (inclui Original) ======================
def scatter_xy(df, xcol, ycol, xlabel, ylabel, fname):
    plt.figure(figsize=(7,5))
    for m in df['modelo'].unique():
        sub = df[df['modelo']==m]
        plt.scatter(sub[xcol], sub[ycol], label=m, alpha=0.7)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel} (inclui Original)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"resultados2/scatter_{fname}.png", dpi=160)
    plt.close()

scatter_xy(df_metrics, 'compression_ratio', 'sim_tfidf', 'Compression ratio', 'Similaridade TF-IDF', 'sim_vs_comp_com_original')
scatter_xy(df_metrics, 'tempo_s', 'sim_tfidf', 'Tempo (s)', 'Similaridade TF-IDF', 'sim_vs_tempo_com_original')


# ====================== 7) Comparativos POR TEXTO (incluem Original) ======================
def plot_len_por_texto(df_lens, i):
    sub = df_lens[df_lens['idx_texto']==i].copy()
    # Ordena para mostrar Original primeiro
    order = ['Original'] + [m for m in sub['modelo'].unique() if m != 'Original']
    sub['modelo'] = pd.Categorical(sub['modelo'], order)
    sub = sub.sort_values('modelo')
    xs = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(9,5))
    bars = ax.bar(xs, sub['len_texto'].values)
    ax.bar_label(bars, fmt='%d', padding=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(sub['modelo'].astype(str).tolist(), rotation=15)
    ax.set_ylabel('Tamanho (nº de caracteres)')
    ax.set_title(f"Texto #{i:03d} — Tamanho: Original vs Resumos")
    plt.tight_layout()
    plt.savefig(f"resultados2/por_texto/texto_{i:03d}_tamanho.png", dpi=160)
    plt.close()

def plot_metric_por_texto(df_metrics, i, col, titulo, ylabel, fname_suffix, fmt='{:,.2f}'):
    sub = df_metrics[df_metrics['idx_texto']==i].copy()
    # Ordena para mostrar Original primeiro
    order = ['Original'] + [m for m in sub['modelo'].unique() if m != 'Original']
    sub['modelo'] = pd.Categorical(sub['modelo'], order)
    sub = sub.sort_values('modelo')
    xs = np.arange(len(sub))
    fig, ax = plt.subplots(figsize=(9,5))
    bars = ax.bar(xs, sub[col].values)
    # rótulos
    labels = [fmt.format(v) if np.isfinite(v) else 'NaN' for v in sub[col].values]
    for rect, lab in zip(bars, labels):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(), lab,
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(sub['modelo'].astype(str).tolist(), rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Texto #{i:03d} — {titulo}")
    ymax = np.nanmax(sub[col].values) if len(sub) else 1
    ax.set_ylim(0, ymax*1.1 if ymax>0 else 1)
    plt.tight_layout()
    plt.savefig(f"resultados2/por_texto/texto_{i:03d}_{fname_suffix}.png", dpi=160)
    plt.close()

# Gera comparativos por texto
num_textos = len(noticias)
for i in range(num_textos):
    plot_len_por_texto(df_lens, i)
    plot_metric_por_texto(df_metrics, i, 'sim_tfidf', "Similaridade TF-IDF (Original vs Resumos)", "cosseno", "sim_tfidf")
    plot_metric_por_texto(df_metrics, i, 'jaccard_kw', "Jaccard KW (Original vs Resumos)", "índice (0–1)", "jaccard")
    plot_metric_por_texto(df_metrics, i, 'compression_ratio', "Compression ratio (Original=1)", "|resumo|/|original|", "compression")
    plot_metric_por_texto(df_metrics, i, 'factual_share', "% trechos factuais", "proporção (0–1)", "factual", fmt="{:,.2f}")
    plot_metric_por_texto(df_metrics, i, 'tempo_s', "Tempo de sumarização", "segundos", "tempo", fmt="{:,.2f}")


# ====================== 8) Saída ======================
print("\nArquivos gerados em /resultados2:")
print("- metrics_por_texto_com_baseline.xlsx")
print("- metrics_por_modelo_com_baseline.xlsx")
print("- tempo_medio_com_original.png, sim_tfidf_medio_com_original.png, jaccard_kw_medio_com_original.png, factual_share_medio_com_original.png, compression_ratio_medio_com_original.png")
print("- radar_modelos.png (sem baseline) e radar_com_original.png (com baseline)")
print("- box_*_com_original.png (todas as métricas, incluindo Original)")
print("- scatter_sim_vs_comp_com_original.png, scatter_sim_vs_tempo_com_original.png")
print("- Pasta /por_texto com gráficos por texto: *_tamanho.png, *_sim_tfidf.png, *_jaccard.png, *_compression.png, *_factual.png, *_tempo.png")
if (agg['modelo']=='Original').sum() == 0:
    print("[AVISO] Baseline 'Original' não incluso; verifique montagem do DataFrame.")
