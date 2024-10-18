import re
from transformers import pipeline

# Lista de conjunções
conjuncao_pattern = r'\b(e|nem|também|bem como|não só...mas|também|mas|porém|contudo|todavia|entretanto|no entanto|não obstante|ou|ou...ou|já...já|ora...ora|quer...quer|seja...seja|logo|pois|portanto|assim|por isso|por consequência|por conseguinte|porque|que|porquanto|visto que|uma vez que|já que|pois que|como|tanto que|tão que|tal que|tamanho que|de forma que|de modo que|de sorte que|de tal forma que|a fim de que|para que|quando|enquanto|agora que|logo que|desde que|assim que|tanto que|apenas|se|caso|desde|salvo se|exceto se|contando que|embora|conquanto|ainda que|mesmo que|se bem que|posto que|assim como|tal|qual|tanto como|conforme|consoante|segundo|à proporção que|à medida que|ao passo que|quanto mais...mais)\b|\.'

# Frase a ser analisada
frase = "Postagens nas redes sociais afirmam falsamente que o TSE não testou a segurança das urnas eletrônicas para as eleições de 2022 e que as vulnerabilidades encontradas em 2012 não foram corrigidas. Essas alegações se baseiam em um vídeo antigo em que um especialista em segurança digital apresenta um projeto desativado."

# 1. Usar regex para separar a frase nas conjunções
partes = re.split(conjuncao_pattern, frase)

# 2. Remover partes vazias e limpar espaços, garantindo que não haja None
partes = [parte.strip() for parte in partes if parte is not None and parte.strip()]

# 3. Exibir as partes separadas
print("Frase separada por conjunções e pontos:")
for i, parte in enumerate(partes):
    print(f"{i + 1}: {parte}")


checkpoint = "Sami92/XLM-R-Large-ClaimDetection"
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
claimdetection = pipeline("text-classification", model = checkpoint, tokenizer =checkpoint, **tokenizer_kwargs, device="cuda")
results = claimdetection(partes)

# Iterando sobre os resultados e printando as saídas
for i, result in enumerate(results):
    if result['label'] == 'factual':
        print(f"Texto {i+1}: {partes[i]}")
        print(f"Resultado: {result}\n")
