# -*- coding: utf-8 -*-
"""
@Author: Eleven
@Date  : 2019/3/28 9:17 AM
"""
from gensim.models import Word2Vec
import Constants as Constants
import numpy as np
import pickle
import re
from nltk.stem.porter import PorterStemmer
porterstemmer = PorterStemmer()

"""数据集"""
file_train=open('/home/sjq/data/FB15K/FB15k/train_text.txt','r',encoding='utf-8')
file_test=open('/home/sjq/data/FB15K/FB15k/test_text.txt','r',encoding='utf-8')
file_valid=open('/home/sjq/data/FB15K/FB15k/valid_text.txt','r',encoding='utf-8')

"""word2vec"""
model = Word2Vec.load('word2vec/w2v_256.model')
# embedding矩阵
word2vec_matrix=model.wv.vectors
print(word2vec_matrix.shape)
# w2v matrix 添加三行作为PAD、ER、RE的向量
word2vec_matrix=np.insert(word2vec_matrix,0,values=np.zeros((3,word2vec_matrix.shape[1])),axis=0)
print(word2vec_matrix.shape)
# 词典，包含index，频次信息
vocab=model.wv.vocab
# 词列表，顺序与矩阵下标一致
index2word=model.wv.index2word
index2word.insert(0,Constants.PAD_WORD)
index2word.insert(1,Constants.ER_WORD)
index2word.insert(2,Constants.RE_WORD)
# 73538
print(len(index2word))
word2index={n:m for m,n in enumerate(index2word)}
print('load word2vec done')

"""FB15K entity"""
mid2name={}
file_fb15k_entity2name = open('/home/sjq/data/FB15K/description/mid2name.txt', 'r', encoding='utf-8')
for line in file_fb15k_entity2name:
    sp = line.strip().split('\t')
    name=re.sub("_+",' ',sp[1]).lower()
    mid2name[sp[0]]=name
file_fb15k_entity2name.close()
print('load entity names done')

entity2id={}
id2entity=[]
file_entity2id=open('/home/sjq/data/FB15K/FB15k/entity2id.txt', 'r', encoding='utf-8')
for line in file_entity2id:
    sp = line.strip().split('\t')
    id2entity.append(mid2name[sp[0]])
    entity2id[mid2name[sp[0]]]=int(sp[1])
file_entity2id.close()
print('load entity ids done')

"""FB15K relation"""
relation2id={}
id2relation=[]
file_relation2id=open('/home/sjq/data/FB15K/entity_word/relation2id.txt', 'r', encoding='utf-8')
for line in file_relation2id:
    sp = line.strip().split('\t')
    id2relation.append(sp[0])
    relation2id[sp[0]]=int(sp[1])
file_relation2id.close()
print('load relation ids done')

entity2count={}
relation2count={}
type2count={}

entity_id2words_id={}
relation_id2words_id={}

triple_pool=set()
entity_pool=set()
relation_pool=set()

max_entity_len=0
max_entity=[]
max_len=0
min_len=50
total_len=0
count=0
"""处理train_text和test_text和valid_text文件"""
def process(file):
    global max_entity_len
    global max_entity
    global max_len
    global min_len
    global total_len
    global count
    sentences_words_id = []
    triples_words_id = []
    triples_id=[]
    triples_len=[]
    ccc=0
    for line in file:
        ccc+=1
        sp = line.strip().split('\t')
        # 三元组(str)
        sub=sp[0]
        pre=id2relation[int(sp[1])]
        obj=sp[2]

        # 统计实体和文本的最大长度
        text_len=len(sp[-1].split())

        if text_len>max_len:
            max_len=text_len
        if text_len<min_len:
            min_len=text_len
        if text_len>200:
            continue
        total_len+=text_len
        count+=1

        if len(sub.split())>max_entity_len:
            max_entity_len=len(sub.split())
            max_entity=sub
        if len(obj.split())>max_entity_len:
            max_entity_len=len(obj.split())
            max_entity=obj

        pre_words=[m.strip('.').split('_') for m in pre.split('/')]
        pre_words_new=''
        for p in pre_words[::-1]:
            flag1=True
            for pp in p:
                if pp not in word2index:
                    flag1=False
                    break
            if not flag1:
                flag1=True
                for pp in p:
                    if porterstemmer.stem(pp) not in word2index:
                        flag1 = False
                        break
                if flag1:
                    pre_words_new=[porterstemmer.stem(pp) for pp in p]
                    break
            else:
                pre_words_new=p
                break
        if pre_words_new=='':
            print('pre_words_new error')

        # 文本(str)
        sentences_words_id.append([word2index[m] for m in sp[-1].split()])
        sub_words_id=[word2index[m] for m in sub.split()]
        pre_words_id=[word2index[m] for m in pre_words_new]
        obj_words_id=[word2index[m] for m in obj.split()]
        if entity2id[sub] not in entity_id2words_id:
            entity_id2words_id[entity2id[sub]]=sub_words_id
        if entity2id[obj] not in entity_id2words_id:
            entity_id2words_id[entity2id[obj]]=obj_words_id
        if relation2id[pre] not in relation_id2words_id:
            relation_id2words_id[relation2id[pre]]=pre_words_id
        triples_words_id.append([sub_words_id,pre_words_id,obj_words_id])
        triples_id.append([entity2id[sub], int(sp[1]), entity2id[obj]])
        triples_len.append([len(m) for m in triples_words_id[-1]])

        if (entity2id[sub], int(sp[1]), entity2id[obj]) not in triple_pool:
            triple_pool.add((entity2id[sub], int(sp[1]), entity2id[obj]))
        if entity2id[sub] not in entity_pool:
            entity_pool.add(entity2id[sub])
        if entity2id[obj] not in entity_pool:
            entity_pool.add(entity2id[obj])
        if int(sp[1]) not in relation_pool:
            relation_pool.add(int(sp[1]))
        # 关系 和对应的 频次
        if pre in relation2count:
            relation2count[pre] += 1
        else:
            relation2count[pre] = 1

        # 实体和实体对应的频次
        if sub in entity2count:
            entity2count[sub] += 1
        else:
            entity2count[sub] = 1

        if obj in entity2count:
            entity2count[obj] += 1
        else:
            entity2count[obj] = 1
    return sentences_words_id, triples_words_id, triples_id, triples_len


train_sen_words_id,train_tri_words_id,train_tri_id,train_tri_len=process(file_train)
test_sen_words_id,test_tri_words_id,test_tri_id,test_tri_len=process(file_test)
valid_sen_words_id,valid_tri_words_id,valid_tri_id,valid_tri_len=process(file_valid)

# 158712 158712 158712 158712
# 19416 19416 19416 19416
# 16642 16642 16642 16642
# 限定text长度小于等于200
# 158137 158137 158137 158137
# 19352 19352 19352 19352
# 16592 16592 16592 16592
print(len(train_sen_words_id),len(train_tri_words_id),len(train_tri_id),len(train_tri_len))
print(len(test_sen_words_id),len(test_tri_words_id),len(test_tri_id),len(test_tri_len))
print(len(valid_sen_words_id),len(valid_tri_words_id),len(valid_tri_id),len(valid_tri_len))


dict2pickle={
            'dict':{"emb_matrix":word2vec_matrix,
                    'word2index':word2index,
                    'index2word':index2word},
            'kb':{
                'entity2id':entity2id,
                'id2entity':id2entity,
                'entity_id2words_id':entity_id2words_id,
                'relation2id':relation2id,
                'id2relation':id2relation,
                'relation_id2words_id':relation_id2words_id,
                'triple_pool':triple_pool,
                'entity_pool':entity_pool,
                'relation_pool':relation_pool},
            'train':{
                'sen_id':train_sen_words_id,
                'triple_words_id':train_tri_words_id,
                'triple_id':train_tri_id,
                'triple_len':train_tri_len},
            'test':{
                'sen_id':test_sen_words_id,
                'triple_words_id':test_tri_words_id,
                'triple_id':test_tri_id,
                'triple_len':test_tri_len}
             }
pickle.dump(dict2pickle,open('all_data.pkl','wb'))

# 最长：200
# 平均：35.87678855735492
print(max_len,min_len)
print(total_len/count)
# 17
# screen actors guild award for outstanding performance by a female actor in a miniseries or television movie
print(max_entity_len)
print(max_entity)
# 14662 13145
# 1345 1209
print(len(entity2id),len(entity_pool))
print(len(relation2id),len(relation_pool))

