from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file=open('/home/sjq/data/FB15K/FB15k/all_text.txt','r',encoding='utf-8')

sentences=[]
for line in file:
    sentences.append(line.strip().split())


model = Word2Vec(sentences, sg=1, size=256,  window=5,  min_count=1,  negative=10, sample=0.001, hs=0, workers=5)
model.save('word2vec/w2v_256.model')
v=model.wv.vocab
print(type(v))
#73535
print(len(v))
t=model.wv
print(type(t))