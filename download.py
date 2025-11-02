import nltk

# Garante que nenhuma versão antiga atrapalhe
nltk.data.path = [
    "C:/Users/carlo/nltk_data",
    "C:/Users/carlo/AppData/Roaming/nltk_data"
]

# Força o download
nltk.download('mac_morpho', download_dir=nltk.data.path[0], force=True)

# Verifica se foi baixado corretamente
from nltk.corpus import mac_morpho
sents = mac_morpho.tagged_sents()
print(f"Sentenças carregadas: {len(sents)}")
