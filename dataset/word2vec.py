from gensim.models import Word2Vec
import os
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file_train=open('my_seg/train_data.txt','r',encoding='utf-8')
file_dev=open('my_seg/dev_data.txt','r',encoding='utf-8')

sentences=[]
for line in file_train:
    j=json.loads(line)
    sentences.append(j['word'])
for line in file_dev:
    j=json.loads(line)
    sentences.append(j['word'])

model = Word2Vec(sentences, sg=1, size=256,  window=5,  min_count=1,  negative=10, sample=0.001, hs=0, workers=5)
model.save('word2vec/w2v_256.model')
v=model.wv.vocab
print(type(v))
print(len(v))
t=model.wv
print(type(t))

