import re

CONJUNCAO_PATTERN = r'\b(e|nem|também|bem como|não só...mas|também|mas|porém|contudo|todavia|entretanto|no entanto|não obstante|ou|ou...ou|já...já|ora...ora|quer...quer|seja...seja|logo|pois|portanto|assim|por isso|por consequência|por conseguinte|porque|que|porquanto|visto que|uma vez que|já que|pois que|como|tanto que|tão que|tal que|tamanho que|de forma que|de modo que|de sorte que|de tal forma que|a fim de que|para que|quando|enquanto|agora que|logo que|desde que|assim que|tanto que|apenas|se|caso|desde|salvo se|exceto se|contando que|embora|conquanto|ainda que|mesmo que|se bem que|posto que|assim como|tal|qual|tanto como|conforme|consoante|segundo|à proporção que|à medida que|ao passo que|quanto mais...mais)\b|\.'

def extrair_trechos_depois_de_que(texto: str) -> list[str]:
    """
    Encontra todos os trechos que começam após a palavra 'que' e
    vão até a próxima conjunção do padrão CONJUNCAO_PATTERN ou até um ponto final.

    Args:
        texto: Texto de entrada.
        incluir_que: Se True, inclui a palavra 'que' no início de cada trecho retornado.

    Returns:
        Lista de strings com os trechos encontrados (sem vazios).
    """
    # Regex: 'que' (palavra), depois capturamos minimamente até a próxima conjunção ou ponto.
    padrao = re.compile(
        rf"\bque\b\s*(?P<trecho>.*?)(?=\s*(?:{CONJUNCAO_PATTERN}))",
        flags=re.IGNORECASE | re.DOTALL,
    )

    resultados = []
    for m in padrao.finditer(texto):
        trecho = m.group("trecho").strip()
        if trecho:  # ignora capturas vazias (ex.: 'que que ...')
            resultados.append(trecho)
    return resultados

with open('textos.txt', 'r') as arquivo:
    noticias = [linha.strip() for linha in arquivo.readlines()]

for noticia in noticias:
    trechos = extrair_trechos_depois_de_que(noticia)
    print(f"Trechos extraídos: {trechos}\n")
    break  # Remova este break para processar todas as notícias no arquivo