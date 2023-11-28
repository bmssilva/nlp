import csv
from transformers import MarianMTModel, MarianTokenizer
# tradução

import langid

## Efetua a tradução usando o modelo pré-treinado

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Carregando o modelo treinado anteriormente
model = 'tradutor_modelo_en_pt'
tokenizer = 'tradutor_tokenizer_en_pt'

loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model)
loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer)

tokenizer_idiom = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model_idiom = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")


import nltk
nltk.download('punkt')
import torch
from torch import optim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_mt_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE").to(device)
tokenizer_mt_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")

def identificar_idioma(frase):
    #frase = "I want to buy car"
    input_ids = loaded_tokenizer(frase, return_tensors="pt").input_ids
    output_ids = loaded_model.generate(input_ids)[0]
    target = loaded_tokenizer.decode(output_ids)
    lang, _ = langid.classify(target)
    return lang

# le o arquivo de acervos
with open('iso_acervos_teste.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';', quotechar='\"')
        corpus = list(reader)
        header, corpus = corpus[0], corpus[1:]

def traduz_en_pt(frase):
    # sentenças a serem traduzidas
    batch_input_str = ((">>pt_br<< "+frase[0]))
    # tokenizando as sentenças
    encoded = loaded_tokenizer(batch_input_str, return_tensors='pt', padding=True)
    # traduzindo
    translated = loaded_model.generate(**encoded)
    # preparando a saída
    decoded_translations = loaded_tokenizer.batch_decode(translated, skip_special_tokens=True)
  
    return decoded_translations[0]
    # Imprimir as traduções
    #for original, traducao in zip(batch_input_str, decoded_translations):
    #    print(f"Original: {original}")
    #    print(f"Tradução: {traducao}\n")

for linha in corpus:
    titulo = [linha[1]]
    idioma = identificar_idioma(titulo)
    #print(f"O titulo {titulo} está no idioma: {idioma}")
    if idioma != "pt" :
        linha[1] = traduz_en_pt(titulo)
        

print(corpus[0])

#print("en->de")
#print("source:", source)
#print("target:", target)



# Exemplo de uso
#frase = "Hello, how are you?"
#idioma = identificar_idioma(target)


#import nltk
#nltk.download('punkt')
#import torch
#from torch import optim
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE").to(device)
#tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ROMANCE")
#source = "I want to buy car"
#input_ids = tokenizer(source, return_tensors="pt").input_ids
#output_ids = model.generate(input_ids)[0]
#target = tokenizer.decode(output_ids)
#print("en->pt")
#print("source:", source)
#print("target:", target)



