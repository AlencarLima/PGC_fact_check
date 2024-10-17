from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException

import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

def configurar_chromedriver():
    """ Função para configurar o ChromeDriver

    Returns:
        _type_: _description_
    """    
    service = Service("C:/Users/carlo/AppData/Local/Programs/Python/Python312/chromedriver.exe")
    chrome_options = Options()
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--headless")
    return service, chrome_options


def ler_links_arquivo(file_name):
    """ Função para ler os links de um arquivo Excel

    Args:
        file_name (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df = pd.read_excel(file_name, engine='openpyxl')
    return df.iloc[0:, 1].tolist()

def extrair_conteudo_links(links):
    """ Função para extrair conteúdo dos links

    Args:
        links ([string]): _description_

    Returns:
        _type_: _description_
    """    
    textos = []
    for link in links:
        try:
            response = requests.get(link)
            response.raise_for_status()
            textos.append(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Erro ao acessar {link}: {e}")
    return textos

def extrair_texto_selenium(url):
    """ Função para extrair texto de uma URL usando Selenium e chromedriver

    Args:
        url (string): link para notícia

    Returns:
        _type_: _description_
    """    
    service, chrome_options = configurar_chromedriver()

    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'span'))
        )
    except TimeoutException:
        print("A página não carregou a tempo.")
        driver.quit()
        return None
    
    html = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(html, 'html.parser')
    text = ''

    for paragraph in soup.find_all('span'):
        text += paragraph.get_text() + ' '
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    
    return text

def processar_texto_html(texto, link=None):
    """Função para processar e limpar o conteúdo HTML

    Args:
        texto (_type_): _description_
        link (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    soup = BeautifulSoup(texto, 'html.parser')
    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.get_text() + ' '

    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()

    # se não houver texto extraído, tenta com Selenium
    if not text and link:
        text = extrair_texto_selenium(link)
    return text

file_name = 'docs_rafael.xlsx'

links = ler_links_arquivo(file_name)
textos = extrair_conteudo_links(links)

noticias = []
for index, texto in enumerate(textos):
    text = processar_texto_html(texto, link=links[index])
    noticias.append(text)

with open('textos.txt', 'w') as arquivo:
    for string in noticias:
        arquivo.write(string + '\n')

with open('textos.txt', 'r') as arquivo:
    # Lendo todas as linhas e removendo o caractere de nova linha '\n'
    vetor = [linha.strip() for linha in arquivo.readlines()]

print(vetor)
