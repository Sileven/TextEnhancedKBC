import json
import pickle
import re

"""文件格式：sub和obj是分词list，pre是词"""
file_known_train=open('/home/sjq/data/FB15K/FB15k/train.txt','r',encoding='utf-8')
file_known_test=open('/home/sjq/data/FB15K/FB15k/test.txt','r',encoding='utf-8')

all_data=pickle.load(open('../all_data.pkl','rb'))
name2id=all_data['kb']['entity2id']
relation2id=all_data['kb']['relation2id']

mid2name={}
file_fb15k_entity2name = open('/home/sjq/data/FB15K/description/mid2name.txt', 'r', encoding='utf-8')
for line in file_fb15k_entity2name:
    sp = line.strip().split('\t')
    name=re.sub("_+",' ',sp[1]).lower()
    mid2name[sp[0]]=name
file_fb15k_entity2name.close()
print('load entity names done')

relation2head={}
relation2tail={}


def process(file):
    for line in file:
        sp=line.strip().split('\t')
        sub_id = name2id[mid2name[sp[0]]]
        pre_id = relation2id[sp[2]]
        obj_id = name2id[mid2name[sp[1]]]

        if pre_id not in relation2head:
            relation2head[pre_id]=[]
            relation2tail[pre_id]=[]
        relation2head[pre_id].append(sub_id)
        relation2tail[pre_id].append(obj_id)


process(file_known_test)
process(file_known_train)

pickle.dump({'relation2head':relation2head,'relation2tail':relation2tail},open('relation2entity.pkl','wb'))