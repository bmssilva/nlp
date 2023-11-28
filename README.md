# Tradução de títulos bibliográficos e classificação de títulos em assunto utilizando transformer

### Ferramentas e modelos
> HuggingFace (https://huggingface.co/)
> 
> BertImbau
> 
### Metodologia utilizada
<img width="402" alt="metodologia" src="https://github.com/bmssilva/nlp/assets/3011040/1bafe1d7-c0ef-4c0f-a1a0-32567f4ec1c0">

## Passos iniciais antes de executar o código

### Passo 1 - efetuar o download do corpus
> Efetuar o download do corpus do tedtalks de tradução do Inglês para o Protuguês, disponivel em https://object.pouta.csc.fi/OPUS-TED2020/v1/tmx/en-pt_br.tmx.gz
> 
> Descompactar o arquivo en-pt_br.tmx.gz para gerar a pasta en-pt_br.tmx
> 
> Descompactar o arquivo dados_iso_acervos_id_assunto.zip
>   

### Passo 2 - instalar as bibliotecas necessárias 

> pip install transformers
> pip install sacremoses

>> Para detecção de idioma foi utilizado o modelo pré-treinado langid que é baseado no modelo lid-mb-3 (LID - Language Identification).
> pip install langid 


### Passo 3 - Execução
Ao executar o python .\pretraining_tradutor_en_pt.py
O script irá ler o corpus da pasta en-pt_br.tmx, em seguida efetuará o treinamento, quando o treinamento finalizar, será gerado duas pastas: tradutor_modelo_en_pt e tradutor_tokenizer_en_pt, nesta pasta está salvo o melhor modelo que foi obtido.

Para usar o modelo gerado, executar o python .\traducao.py, este lê o modelo gerado, e executa a tradução do arquivo de teste iso_acervos_teste.csv

## Referências
Este código foi implementado com base no tutotrial disponível em https://www.youtube.com/watch?v=GncyWR-dYW8&list=PLLrlHSmC0Mw73a1t73DEjgGMPyu8QssWT&index=51 disponibilizado neste canal https://www.youtube.com/@Thicasfer

Também é possivel acessar a versão no COLAB, disponível em https://colab.research.google.com/drive/1oVc-snXASxmlks2fHjxZ81fKG_mBW-jR?usp=sharing

Dados dos títulos e assuntos dos acervos, foram extraidos da biblioteca central da PUCPR - Pontifícia Universidade Católica do Paraná https://pergamum-biblioteca.pucpr.br/
