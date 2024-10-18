from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Carregar o modelo BERT e o tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Função para tokenizar e gerar embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    # Pegar os embeddings da última camada
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0]  # Ignorar batch dimension

# Função para segmentar o texto com base em similaridade
def segment_text(text):
    embeddings = get_bert_embeddings(text)
    similarities = []
    
    # Calcular similaridade entre embeddings consecutivos
    for i in range(len(embeddings) - 1):
        sim = torch.cosine_similarity(embeddings[i], embeddings[i + 1], dim=0)
        similarities.append(sim.item())
    
    # Identificar quebras onde a similaridade é baixa
    threshold = np.mean(similarities) - np.std(similarities)
    split_points = [i for i, sim in enumerate(similarities) if sim < threshold]

    # Usar os pontos de quebra para segmentar o texto
    tokens = tokenizer.tokenize(text)
    sentences = []
    start = 0
    for point in split_points:
        sentence = tokenizer.convert_tokens_to_string(tokens[start:point + 1])
        sentences.append(sentence)
        start = point + 1

    # Adicionar a última sentença
    if start < len(tokens):
        sentences.append(tokenizer.convert_tokens_to_string(tokens[start:]))

    return sentences

# Testar com um exemplo
texto = """Postagens nas redes sociais afirmam falsamente que o TSE não testou a segurança das urnas eletrônicas para as eleições de 2022 e que as vulnerabilidades encontradas em 2012 não foram corrigidas. Essas alegações se baseiam em um vídeo antigo em que um especialista em segurança digital apresenta um projeto desativado."""
sentences = segment_text(texto)

# Imprimir as sentenças
reconstructed_sentences = []
for i, sentence in enumerate(sentences):
    if i > 0 and sentence.startswith("##"):
        # Juntar com a sentença anterior
        reconstructed_sentences[-1] += sentence[2:]  # Remove '##' e junta
    else:
        reconstructed_sentences.append(sentence)

# Exibir as sentenças reconstruídas
for i, sentence in enumerate(reconstructed_sentences):
    print(f"Sentença {i + 1}: {sentence}")

from transformers import pipeline

checkpoint = "Sami92/XLM-R-Large-ClaimDetection"
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
claimdetection = pipeline("text-classification", model = checkpoint, tokenizer =checkpoint, **tokenizer_kwargs, device="cuda")
results = claimdetection(reconstructed_sentences)

# Iterando sobre os resultados e printando as saídas
for i, result in enumerate(results):
    if result['label'] == 'factual':
        print(f"Texto {i+1}: {reconstructed_sentences[i]}")
        print(f"Resultado: {result['score']}\n")
