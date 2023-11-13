import csv
from random import shuffle
import nltk
nltk.download('punkt')

import os
import sys

import torch
import torch.nn as nn
from torch import optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torch.utils.data import DataLoader

with open('iso.csv', 'r', encoding='utf-8') as f:
  reader = csv.reader(f, delimiter=';', quotechar='\"')
  corpus = list(reader)

  header, corpus = corpus[0], corpus[1:]

corpus = corpus[:10000]
shuffle(corpus)
print(len(corpus)) 
#print(reader) 
#i=0
#for w in corpus:
#    i=i+1
#    if len(w) >= 4:
#        print(i, len(w), w )
#        print(w[4])


titulo = [w[1] for w in corpus if len(w[1]) > 1]
assunto = [2 if w[4] in ['Antropologia', 'Aculturação'] else 0 if w[4] in ['Administração', 'administrativo'] else 1 for w in corpus if len(w) >= 4]
data = [{ 'X': titulo, 'y': assunto } for (titulo, assunto) in zip(titulo, assunto)]


#reviews = [w[0][10] for w in corpus]
#ratings = [2 if w[0][8] in ['4', '5'] else 0 if w[0][8] in ['1', '2'] else 1 for w in corpus]
#data = [{ 'X': review, 'y': rating } for (review, rating) in zip(reviews, ratings)]

#print(titulo)
print(len(data))
# Separa os dados em conjunto de treino e teste
size = int(len(data) * 0.2)
treino = data[size:]
teste = data[:size]

print(len(treino), len(teste))

# Instanciando parâmetros

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nclasses = 3 # negativo, neutra, positiva
nepochs = 2 # épocas de treino
batch_size = 8 # tamanho dos lotes
batch_status = 32 # 
learning_rate = 1e-5 # taxa de aprendizado 5 casas decimas
early_stop = 2 # se em 2 épocas consecutivas o resultando não melhorar no conjunto de teste para o treinamento

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
  
  print(classification_report(y_real, y_pred, labels=[0, 1, 2], target_names=['Administração', 'Sem tag assunto', 'Antropologia']))
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

