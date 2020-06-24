# -*- coding: utf-8 -*-
"""
@Author: Eleven
@Date  : 2019/3/28 9:17 AM
"""
import json
from gensim.models import Word2Vec
import jieba
import Constants as Constants
import numpy as np
jieba.load_userdict('dict/user_dict.txt')

import pickle

zh_punc = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
en_punc="!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
"""文件格式：sub和obj是分词list，pre是词"""
file_known_train=open('split/known_train.json','r',encoding='utf-8')
file_known_test=open('split/known_test.json','r',encoding='utf-8')

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

# 292928
print(len(index2word))

word2index={n:m for m,n in enumerate(index2word)}

entity2count={}
relation2count={}
type2count={}

# 类型名->实体名set
type2entity={}
# 实体名->实体词list
entity2words={}
relation2words={}
# 实体名->类型名
entity2type={}
# max_entity_len=0
# max_entity=[]
"""处理known_train和known_test文件"""
def process(file):
    global max_entity_len
    global max_entity
    sentences = []
    triples_words = []
    types=[]
    triple_len=[]
    ccc=0
    for line in file:
        ccc+=1
        j = json.loads(line)
        spo=j['spo']
        # if len(spo['subject'])>max_entity_len and len(spo['subject'])!=25:
        #     max_entity_len=len(spo['subject'])
        #     max_entity=spo['subject']
        # if len(spo['object'])>max_entity_len and len(spo['object'])!=25:
        #     max_entity_len=len(spo['object'])
        #     max_entity=spo['object']
        sub=''.join(spo['subject'])
        pre=spo['predicate']
        obj=''.join(spo['object'])
        sub_type=spo['subject_type']
        obj_type=spo['object_type']

        if sub in entity2words:
            if entity2words[sub]!=spo['subject']:
                print('error')
                print(ccc,entity2words[sub],spo['subject'])
        else:
            entity2words[sub]=spo['subject']
            entity2type[sub]=sub_type

        if obj in entity2words:
            if entity2words[obj]!=spo['object']:
                print('error')
                print(ccc,entity2words[obj],spo['object'])
        else:
            entity2words[obj]=spo['object']
            entity2type[obj]=obj_type

        # 分别存下句子，三元组(词切分)，头为实体类型
        sentences.append(j['word'])
        #print(pre)
        if pre in word2index:
            pre_list=[pre]
        else:
            if pre == '人口数量':
                pre_list=[pre[0:2],pre[2:]]
            else:
                pre_list=[]
                pre_cut=jieba.lcut(pre)
                for ss in pre_cut:
                    if ss in word2index:
                        pre_list.append(ss)
                    else:
                        print(pre,'error')
                        exit(0)
        if pre not in relation2words:
            relation2words[pre]=pre_list
        triples_words.append([spo['subject'], pre_list, spo['object']])
        types.append([sub_type,obj_type])
        triple_len.append([len(spo['subject']),len(pre_list),len(spo['object'])])

        # 记录所有实体类型对应的实体、实体类型和对应的频次
        if sub_type not in type2entity:
            type2entity[sub_type]=set([sub])
            type2count[sub_type]=1
        else:
            type2entity[sub_type].add(sub)
            type2count[sub_type]+=1
        if obj_type not in type2entity:
            type2entity[obj_type]=set([obj])
            type2count[obj_type]=1
        else:
            type2entity[obj_type].add(obj)
            type2count[obj_type]+=1

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
    return sentences,triples_words,types,triple_len


sen_train,tri_words_train,types_train,triple_len_train=process(file_known_train)
sen_test,tri_words_test,types_test,triple_len_test=process(file_known_test)

"""实体,关系,实体类型转id"""
entity2count_sorted=sorted(entity2count.items(),key=lambda item:item[1],reverse=True)
relation2count_sorted=sorted(relation2count.items(),key=lambda item:item[1],reverse=True)
type2count_sorted=sorted(type2count.items(),key=lambda item:item[1],reverse=True)

entity=[m[0] for m in entity2count_sorted]
entity2id={m:n for n,m in enumerate(entity)}

relation=[m[0] for m in relation2count_sorted]
relation2id={m:n for n,m in enumerate(relation)}

type=[m[0] for m in type2count_sorted]
type2id={m:n for n,m in enumerate(type)}

type_id2entity_id={}
entity_id2words_id={}
relation_id2words_id={}
entity_id2type_id={}

for t,e in type2entity.items():
    # print(e)
    # print(type2id[t])
    type_id2entity_id[type2id[t]]=set([entity2id[m] for m in e])
print(len(type2count))
print(len(type2entity))
print(len(type_id2entity_id))

for e,w in entity2words.items():
    entity_id2words_id[entity2id[e]]=[word2index[m] for m in w]
for e,t in entity2type.items():
    entity_id2type_id[entity2id[e]]=type2id[t]
for r,w in relation2words.items():
    relation_id2words_id[relation2id[r]]=[word2index[m] for m in w]
"""******"""
max_len=0
total_len=0
count=0
def word2id(sen,tri_words,types):
    global max_len
    global total_len
    global count
    sentences_words_id = []
    triples_words_id = []
    triples_id = []
    types_id=[]
    for sen,tri,t in zip(sen,tri_words,types):
        if len(sen)>max_len:
            max_len=len(sen)
        total_len+=len(sen)
        count+=1

        sentences_words_id.append([word2index[m] for m in sen])
        sub_words=[word2index[m] for m in tri[0]]
        obj_words=[word2index[m] for m in tri[2]]
        pre_words=[word2index[m] for m in tri[1]]
        # if tri[1] in word2index:
        #     pre_words=[word2index[tri[1]]]
        # else:
        #     if tri[1] == '人口数量':
        #         pre_words=[word2index[tri[1][0:2]],word2index[tri[1][2:]]]
        #     else:
        #         pre_words=[]
        #         pre_cut=jieba.lcut(tri[1])
        #         for ss in pre_cut:
        #             if ss in word2index:
        #                 pre_words.append(word2index[ss])
        #             else:
        #                 print(tri[1],'error')
        #                 exit(0)
        triples_words_id.append([sub_words,pre_words,obj_words])
        #print(pre_words)
        triples_id.append([entity2id[''.join(tri[0])],relation2id[''.join(tri[1])],entity2id[''.join(tri[2])]])
        types_id.append([type2id[t[0]],type2id[t[1]]])
    return sentences_words_id,triples_words_id,triples_id,types_id


train_sentences_words_id,train_triples_words_id,train_triples_id,train_types_id=word2id(sen_train,tri_words_train,types_train)
test_sentences_words_id,test_triples_words_id,test_triples_id,test_types_id=word2id(sen_test,tri_words_test,types_test)

# train:265789 test:52013
print(len(train_sentences_words_id),len(train_triples_words_id),len(train_triples_id),len(train_types_id))
print(len(test_sentences_words_id),len(test_triples_words_id),len(test_triples_id),len(test_types_id))

dict2pickle={
            'dict':{"emb_matrix":word2vec_matrix,
                    'word2index':word2index,
                    'index2word':index2word},
            'kb':{
                'entity2id':entity2id,
                'id2entity':entity,
                'entity_id2words_id':entity_id2words_id,
                'entity_id2type_id':entity_id2type_id,
                'type_id2entity_id':type_id2entity_id,
                'relation2id':relation2id,
                'id2relation':relation,
                'relation_id2words_id':relation_id2words_id,
                'type2id':type2id,
                'id2type':type},
            'train':{
                'sen_id':train_sentences_words_id,
                'triple_words_id':train_triples_words_id,
                'triple_id':train_triples_id,
                'triple_len':triple_len_train,
                'type_id':train_types_id},
            'test':{
                'sen_id':test_sentences_words_id,
                'triple_words_id':test_triples_words_id,
                'triple_id':test_triples_id,
                'triple_len':triple_len_test,
                'type_id':test_types_id}
             }
pickle.dump(dict2pickle,open('all_data_new_correct.pkl','wb'))

# 最长：202
# 平均：31.928018074146795
print(max_len)
print(total_len/count)

