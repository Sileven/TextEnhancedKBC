import json
import pickle

"""文件格式：sub和obj是分词list，pre是词"""
file_known_train=open('split/known_train.json','r',encoding='utf-8')
file_known_test=open('split/known_test.json','r',encoding='utf-8')

all_data=pickle.load(open('all_data_new_correct.pkl','rb'))
entity2id=all_data['kb']['entity2id']
relation2id=all_data['kb']['relation2id']

relation2head={}
relation2tail={}


def process(file):
    for line in file:
        j = json.loads(line)
        spo=j['spo']
        sub=''.join(spo['subject'])
        pre=spo['predicate']
        obj=''.join(spo['object'])
        sub_id=entity2id[sub]
        pre_id=relation2id[pre]
        obj_id=entity2id[obj]
        if pre_id not in relation2head:
            relation2head[pre_id]=[]
            relation2tail[pre_id]=[]
        relation2head[pre_id].append(sub_id)
        relation2tail[pre_id].append(obj_id)


process(file_known_test)
process(file_known_train)

pickle.dump({'relation2head':relation2head,'relation2tail':relation2tail},open('relation2entity.pkl','wb'))