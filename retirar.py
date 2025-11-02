import pandas as pd

# Caminho do arquivo Excel
arquivo = "./resultados/check_worthy_results.xlsx"

# Lê a planilha (por padrão, a primeira)
df = pd.read_excel(arquivo)

# Garante que a coluna termos_relevantes existe
coluna = 'termos_relevantes'

if coluna in df.columns:
    # Remove [, ], ' e espaços desnecessários
    df[coluna] = df[coluna].astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()

    # Salva em um novo arquivo Excel
    df.to_excel("arquivo_limpo.xlsx", index=False)

    print("✅ Coluna F limpa e salva em 'arquivo_limpo.xlsx'.")
else:
    print(f"⚠️ A coluna '{coluna}' não foi encontrada no arquivo.")
