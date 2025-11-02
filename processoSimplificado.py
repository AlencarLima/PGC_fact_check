from summarizer.sbert import SBertSummarizer
from keybert import KeyBERT
import time
import pandas as pd
import nltk
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import mac_morpho
from nltk.tag import UnigramTagger
from nltk.tag import BigramTagger
from collections import Counter
from transformers import pipeline
import re
from nltk import word_tokenize
from nltk.corpus import stopwords

def ler_links_arquivo(file_name):
    """ Função para ler os links de um arquivo Excel

    Args:
        file_name (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df = pd.read_excel(file_name, engine='openpyxl')
    return df.iloc[0:, 1].tolist()

def calcular_relevancia_texto(noticia):
    """ Função para calcular as sentenças mais relevantes com TF-IDF

    Args:
        noticia (string): noticia para ser analisada

    Returns:
        _type_: _description_
    """    
    vectorizer = TfidfVectorizer()

    sentences = nltk.sent_tokenize(noticia)

    # transforma o texto em vetores TF-IDF
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # calcular a similaridade de cosseno entre as sentenças
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sentence_scores = cosine_similarities.mean(axis=1)

    # retorna 5 sentenças mais relevantes
    top_sentence_indices = heapq.nlargest(5, range(len(sentence_scores)), key=sentence_scores.__getitem__)

    return [sentences[i] for i in top_sentence_indices]

def processar_noticias_e_calcular_relevancia(file_name, modelo):
    """ Função principal para processar todas as notícias e calcular sentenças relevantes

    Args:
        file_name (string): _description_
    """    
    links = ler_links_arquivo(file_name)


    with open('textos.txt', 'r') as arquivo:
        noticias = [linha.strip() for linha in arquivo.readlines()]

    resumos = []
    resumos_bert = []
    for noticia in noticias:
        sentencas_relevantes = calcular_relevancia_texto(noticia)
        resumo = ""
        for i, sentenca in enumerate(sentencas_relevantes):
            if i < len(sentencas_relevantes) - 1:
                resumo += sentenca + " "
            else:
                resumo += sentenca
        resumos.append(resumo)

        
        resumo_bert = modelo(noticia)

        resumos_bert.append(resumo_bert)


    return links, resumos, resumos_bert

def treinar_tagger_portugues():
    treino = mac_morpho.tagged_sents()
    unigram_tagger = UnigramTagger(treino)
    bigram_tagger = BigramTagger(treino, backoff=unigram_tagger)
    return bigram_tagger

def extrair_palavras_chaves(texto, tagger, keyBert):
    tokens = word_tokenize(texto)
    pos_tags = tagger.tag(tokens)
    
    if pos_tags is None:
        print("Erro: pos_tags é None.")
        return []
    
    # carrega stopwords em português
    stop_words = list(stopwords.words('portuguese'))
    
    frases_nominais = []
    frase_atual = []

    for palavra, tag in pos_tags:
        # N: Substantivo comum  NPROP: Substantivo próprio  ADJ: adjetivo
        if tag in ('N', 'NPROP', 'ADJ') and palavra not in stop_words:
            frase_atual.append(palavra)
        else:
            if frase_atual:
                frases_nominais.append(' '.join(frase_atual))
                frase_atual = []

    if frase_atual:
        frases_nominais.append(' '.join(frase_atual))
    
    # extraindo palavras-chave usando contagem de frequência
    contagem = Counter(token for token in tokens if token not in stop_words)
    principais_palavras = contagem.most_common(10)
    
    # extraindo palavras-chave usando TF-IDF
    texto_unido = ' '.join(frases_nominais)
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform([texto_unido])

    tfidf_scores = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out(), columns=["score"])
    principais_tfidf = tfidf_scores.nlargest(10, "score")

    palavras_chaves_bert = keyBert.extract_keywords(texto)

    return frases_nominais, principais_palavras, principais_tfidf.index.tolist(), palavras_chaves_bert


if __name__ == "__main__":
    model = SBertSummarizer('paraphrase-MiniLM-L6-v2')

    kw_model = KeyBERT()

    start_time = time.time()

    file_name = 'docs_rafael.xlsx'

    links, resumos, resumos_bert = processar_noticias_e_calcular_relevancia(file_name, model)

    tagger = treinar_tagger_portugues()

    todos_principais_tfidf = []
    for resumo in resumos:
        frases_nominais, principais_palavras, palavras_chave_tf_idf, palavras_chave_bert = extrair_palavras_chaves(resumo, tagger, kw_model)
        todos_principais_tfidf.append(palavras_chave_tf_idf)

    todos_principais_bert = []
    for resumo in resumos_bert:
        frases_nominais, principais_palavras, palavras_chave_tf_idf, palavras_chave_bert = extrair_palavras_chaves(resumo, tagger, kw_model)
        todos_principais_bert.append(palavras_chave_bert)

    vetor_ajustado = []
    for vetor in todos_principais_bert:
        novo_vetor = []
        for item in vetor:
            novo_vetor.append(item[0])

        string_unica = ", ".join(novo_vetor)
        vetor_ajustado.append(string_unica)

    end_time = time.time()

    checkpoint = "Sami92/XLM-R-Large-ClaimDetection"
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    claimdetection = pipeline("text-classification", model=checkpoint, tokenizer=checkpoint, **tokenizer_kwargs, device="cpu")

    conjuncao_pattern = r'\b(e|nem|também|bem como|não só...mas|também|mas|porém|contudo|todavia|entretanto|no entanto|não obstante|ou|ou...ou|já...já|ora...ora|quer...quer|seja...seja|logo|pois|portanto|assim|por isso|por consequência|por conseguinte|porque|que|porquanto|visto que|uma vez que|já que|pois que|como|tanto que|tão que|tal que|tamanho que|de forma que|de modo que|de sorte que|de tal forma que|a fim de que|para que|quando|enquanto|agora que|logo que|desde que|assim que|tanto que|apenas|se|caso|desde|salvo se|exceto se|contando que|embora|conquanto|ainda que|mesmo que|se bem que|posto que|assim como|tal|qual|tanto como|conforme|consoante|segundo|à proporção que|à medida que|ao passo que|quanto mais...mais)\b|\.'

    termos_relevantes_geral = []
    for resumo in resumos_bert:
        partes = re.split(conjuncao_pattern, resumo)
        partes = [parte.strip() for parte in partes if parte is not None and parte.strip()]
        results = claimdetection(partes)
        termos_relevantes_resumo = []
        for i, result in enumerate(results):
            if result['label'] == 'factual':
                termos_relevantes_resumo.append(partes[i])
        termos_relevantes_geral.append(str(termos_relevantes_resumo))