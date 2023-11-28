import csv
from random import shuffle
import nltk
nltk.download('punkt')

import os
#import sys
#sys.path.append('C:\\Python311\\Lib\\site-packages')

import torch
import torch.nn as nn
from torch import optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torch.utils.data import DataLoader


import csv
from random import shuffle
import nltk
nltk.download('punkt')

import os
#import sys
#sys.path.append('C:\\Python311\\Lib\\site-packages')

import torch
import torch.nn as nn
from torch import optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torch.utils.data import DataLoader



with open('/content/iso_acervos.csv', 'r', encoding='utf-8') as f:
  reader = csv.reader(f, delimiter=';', quotechar='\"')
  corpus = list(reader)

  header, corpus = corpus[0], corpus[1:]

shuffle(corpus) # embaralha
corteBase = 10000 #int(len(corpus)/2)
corpus = corpus[:corteBase]# seleciona apenas os 10.000
print(len(corpus))

# Corpus assunto
with open('/content/id_assunto.csv', 'r', encoding='utf-8') as f:
  readerAssunto = csv.reader(f, delimiter=';', quotechar='\"')
  corpusAssunto = list(readerAssunto)

  readerAssunto, corpusAssunto = corpusAssunto[0], corpusAssunto[1:]

#print(reader)
i=0
assunto_descricao = {}
id_assunto = {}
for linha in corpusAssunto:
    i=i+1
    id_assunto[i] = linha[0].strip()
    assunto_descricao[i] = linha[1].strip()

#print("###############", id_assunto[1], assunto_descricao[1])
# Convertendo os valores do dicionário em uma lista
valores_lista = list(id_assunto.values())

# Encontrando a posição do valor desejado (por exemplo, '106565')
id = int(90519)
id_str = str(id)
posicao = valores_lista.index(id_str)
print(posicao)
print(assunto_descricao[posicao+1])



titulo = [w[1] for w in corpus if len(w) > 2]
assunto = [int]


for w in corpus:
   if len(w) > 2:
      listaAssunto = str(w[len(w)-2])
      arrayAssuntos = listaAssunto.split(" ")
      numeroIdAssunto = int(arrayAssuntos[1])
      posicao = valores_lista.index(str(numeroIdAssunto))
      w[len(w)-2] = posicao

assunto = [w[len(w)-2] for w in corpus if len(w) > 2]

data = [{ 'X': titulo, 'y': assunto } for (titulo, assunto) in zip(titulo, assunto)]


#reviews = [w[0][10] for w in corpus]
#ratings = [2 if w[0][8] in ['4', '5'] else 0 if w[0][8] in ['1', '2'] else 1 for w in corpus]
#data = [{ 'X': review, 'y': rating } for (review, rating) in zip(reviews, ratings)]

print(len(titulo))
print(len(assunto))
print(len(data))
# Separa os dados em conjunto de treino e teste
size = int(len(data) * 0.3)
treino = data[size:]
teste = data[:size]

print(len(treino), len(teste))

# Instanciando parâmetros

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nclasses = max(assunto)+1 # negativo, neutra, positiva
nepochs = 100 # épocas de treino
batch_size = 8 # tamanho dos lotes
batch_status = 32 #
learning_rate = 1e-5 # taxa de aprendizado 5 casas decimas
early_stop = 10 # se em 2 épocas consecutivas o resultando não melhorar no conjunto de teste para o treinamento

max_length = 255 # trucar todas as sequências de tokens com no máximo 180, se tiver um texto com mais de 180 tokens, e feito o trunc para 189
write_path = 'modeliso' # salva os melhores modelos nessa pasta model

# Separando os dados em batches

traindata = DataLoader(treino, batch_size=batch_size, shuffle=True)
testdata = DataLoader(teste, batch_size=batch_size, shuffle=True)

# Inicilizando tokenizador, modelo, função de erro e otimizador:
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained('neuralmind/bert-base-portuguese-cased', num_labels=nclasses).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Definindo método de avaliação
def evaluate(model, testdata):
  model.eval()
  y_real, y_pred = [], []
  for batch_idx, inp in enumerate(testdata):
    texts, labels = inp['X'], inp['y']

    # classifying
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    output = model(**inputs)

    pred_labels = torch.argmax(output.logits, 1)

    y_real.extend(labels.tolist())
    y_pred.extend(pred_labels.tolist())

    if (batch_idx+1) % batch_status == 0:
      print('Progress:', round(batch_idx / len(testdata) if len(testdata) > 0 else batch_idx , 2), batch_idx)

  #print(classification_report(y_real, y_pred, labels=assunto, target_names=[descAssunto[1] for descAssunto in corpusAssunto]))
  f1 = f1_score(y_real, y_pred, average='weighted')
  acc = accuracy_score(y_real, y_pred)
  return f1, acc

# treinamento
max_f1, repeat = 0, 0
for epoch in range(nepochs):
  model.train()
  f1, acc = evaluate(model, testdata)
  losses = []
  for batch_idx, inp in enumerate(traindata):
    texts, labels = inp['X'], inp['y']
    #if len(inp['X']) != len(inp['y']) :
    #  continue
    #print(texts)
    #print(labels)
    # classifying
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)

    output = model(**inputs, labels=labels.to(device))

    # Calculate loss
    loss = output.loss
    losses.append(float(loss))

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Display
    if (batch_idx+1) % batch_status == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTotal Loss: {:.6f}'.format(epoch, \
        batch_idx+1, len(traindata), 100. * batch_idx / len(traindata),
        float(loss), round(sum(losses) / len(losses), 5)))

  f1, acc = evaluate(model, testdata)
  print('F1: ', f1, 'Accuracy: ', acc)
  if f1 > max_f1:
    model.save_pretrained(os.path.join(write_path, 'model'))
    max_f1 = f1
    repeat = 0
    print('Saving best model...')
  else:
    repeat += 1

  if repeat == early_stop:
    break


# Utilizando o modelo

# sentenças a serem traduzidas
batch_input_str = (("O conhecimento dos mundos superiores"))
# tokenizando as sentenças
encoded = tokenizer(batch_input_str, return_tensors='pt', padding=True).to(device)
# gerando assunto
outputs =  model(**encoded)
# Probabilidades de saída
logits = outputs.logits

# Converte as probabilidades para predições
predictions = torch.argmax(logits, dim=1).item()

print(f"Assunto previsto: {predictions}")
print(assunto_descricao[predictions])
# preparando a saída
#tokenizer.batch_decode(assuntoParaTitulo, skip_special_tokens=True)