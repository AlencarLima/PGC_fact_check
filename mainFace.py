from transformers import pipeline

texts = [
    'Postagens nas redes sociais afirmam falsamente',  'TSE não testou a segurança das urnas eletrônicas para as eleições de 2022',  'vulnerabilidades encontradas em 2012 não foram corrigidas.',  'Essas alegações se baseiam em um vídeo antigo' , 'um especialista em segurança digital apresenta um projeto desativado.',
    'teste de coisa aleatoria foi feito'
]
checkpoint = "Sami92/XLM-R-Large-ClaimDetection"
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
claimdetection = pipeline("text-classification", model = checkpoint, tokenizer =checkpoint, **tokenizer_kwargs, device="cuda")
results = claimdetection(texts)

# Iterando sobre os resultados e printando as saídas
for i, result in enumerate(results):
    print(f"Texto {i+1}: {texts[i]}")
    print(f"Resultado: {result}\n")