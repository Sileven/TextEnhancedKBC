import json
import pickle

"""文件格式：sub和obj是分词list，pre是词"""
file_uk=open('split/unknown_uk1.json','r',encoding='utf-8')
file_ku=open('split/unknown_ku1.json','r',encoding='utf-8')
file_uu=open('split/unknown_uu1.json','r',encoding='utf-8')

all_data=pickle.load(open('all_data_new_correct.pkl','rb'))
word2index=all_data['dict']['word2index']
entity2id=all_data['kb']['entity2id']
relation2id=all_data['kb']['relation2id']
entity_id2words_id=all_data['kb']['entity_id2words_id']
relation_id2words_id=all_data['kb']['relation_id2words_id']
type2id=all_data['kb']['type2id']

max_len=0
total_len=0
count=0
def process(file,type):
    global max_len
    global total_len
    global count
    sentences_words_id = []
    triples_words_id = []
    triples_id = []
    triples_len=[]
    types_id=[]

    for line in file:
        j = json.loads(line)
        text=j['word']
        spo=j['spo']
        sub=''.join(spo['subject'])
        pre=spo['predicate']
        obj=''.join(spo['object'])
        sub_type=spo['subject_type']
        obj_type=spo['object_type']

        if len(text)>max_len:
            max_len=len(text)
        total_len+=len(text)
        count+=1

        sentences_words_id.append([word2index[m] for m in text])
        sub_words=[word2index[m] for m in spo['subject']]
        obj_words=[word2index[m] for m in spo['object']]
        pre_words=relation_id2words_id[relation2id[pre]]
        triples_words_id.append([sub_words,pre_words,obj_words])
        if type==1:
            triples_id.append([-1,relation2id[pre],entity2id[obj]])
        elif type==2:
            triples_id.append([entity2id[sub], relation2id[pre], -1])
        else:
            triples_id.append([-1,relation2id[pre],-1])
        triples_len.append([len(sub_words),len(pre_words),len(obj_words)])
        types_id.append([type2id[sub_type],type2id[obj_type]])

    return sentences_words_id, triples_words_id, triples_id, triples_len, types_id


sen_uk,tri_words_uk,tri_uk,tri_len_uk,type_uk=process(file_uk,1)
sen_ku,tri_words_ku,tri_ku,tri_len_ku,type_ku=process(file_ku,2)
sen_uu,tri_words_uu,tri_uu,tri_len_uu,type_uu=process(file_uu,3)

dict2pickle={
            'dict':{"emb_matrix":all_data['dict']['emb_matrix'],
                    'word2index':all_data['dict']['word2index'],
                    'index2word':all_data['dict']['index2word']},
            'kb':{
                'entity2id':all_data['kb']['entity2id'],
                'id2entity':all_data['kb']['id2entity'],
                'entity_id2words_id':all_data['kb']['entity_id2words_id'],
                'entity_id2type_id':all_data['kb']['entity_id2type_id'],
                'type_id2entity_id':all_data['kb']['type_id2entity_id'],
                'relation2id':all_data['kb']['relation2id'],
                'id2relation':all_data['kb']['id2relation'],
                'relation_id2words_id':all_data['kb']['relation_id2words_id'],
                'type2id':all_data['kb']['type2id'],
                'id2type':all_data['kb']['id2type']},
            'test_uk':{
                'sen_id':sen_uk,
                'triple_words_id':tri_words_uk,
                'triple_id':tri_uk,
                'triple_len':tri_len_uk,
                'type_id':type_uk},
            'test_ku':{
                'sen_id':sen_ku,
                'triple_words_id':tri_words_ku,
                'triple_id':tri_ku,
                'triple_len':tri_len_ku,
                'type_id':type_ku},
            'test_uu':{
                'sen_id':sen_uu,
                'triple_words_id':tri_words_uu,
                'triple_id':tri_uu,
                'triple_len':tri_len_uu,
                'type_id':type_uu},
             }
pickle.dump(dict2pickle,open('all_data_for_new.pkl','wb'))

# 最长：199
# 平均：31.173600989355595
print(max_len)
print(total_len/count)