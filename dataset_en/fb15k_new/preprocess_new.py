import json
import pickle
from nltk.stem.porter import PorterStemmer
porterstemmer = PorterStemmer()
"""文件格式：sub和obj是分词list，pre是词"""
file_uk=open('/home/sjq/data/FB15K/FB20K-new/text_uk.txt','r',encoding='utf-8')
file_ku=open('/home/sjq/data/FB15K/FB20K-new/text_ku.txt','r',encoding='utf-8')
file_uu=open('/home/sjq/data/FB15K/FB20K-new/text_uu.txt','r',encoding='utf-8')

all_data=pickle.load(open('../all_data.pkl','rb'))
word2index=all_data['dict']['word2index']
entity2id=all_data['kb']['entity2id']
id2relation=all_data['kb']['id2relation']
entity_id2words_id=all_data['kb']['entity_id2words_id']
relation_id2words_id=all_data['kb']['relation_id2words_id']

max_len=0
min_len=0
max_entity_len=0
total_len=0
count=0
max_entity=[]

def process(file,type):
    global max_len
    global min_len
    global max_entity_len
    global total_len
    global count
    global max_entity
    sentences_words_id = []
    triples_words_id = []
    triples_id = []
    triples_len=[]

    for line in file:
        sp=line.split('\t')

        sub = sp[0]
        pre = id2relation[int(sp[1])]
        obj = sp[2]

        if len(sub.split()) > max_entity_len:
            max_entity_len = len(sub.split())
            max_entity = sub
        if len(obj.split()) > max_entity_len:
            max_entity_len = len(obj.split())
            max_entity = obj

        pre_words = [m.strip('.').split('_') for m in pre.split('/')]
        pre_words_new = ''
        for p in pre_words[::-1]:
            flag1 = True
            for pp in p:
                if pp not in word2index:
                    flag1 = False
                    break
            if not flag1:
                flag1 = True
                for pp in p:
                    if porterstemmer.stem(pp) not in word2index:
                        flag1 = False
                        break
                if flag1:
                    pre_words_new = [porterstemmer.stem(pp) for pp in p]
                    break
            else:
                pre_words_new = p
                break
        if pre_words_new == '':
            print('pre_words_new error')

        words_id_list=[word2index[m] for m in sp[-1].split() if m in word2index]

        # 统计实体和文本的最大长度
        text_len = len(words_id_list)

        if text_len > max_len:
            max_len = text_len
        if text_len < min_len:
            min_len = text_len
        if text_len > 200:
            continue
        total_len += text_len
        count += 1

        sentences_words_id.append(words_id_list)

        sub_words=[word2index[m] for m in sub.split()]
        pre_words=[word2index[m] for m in pre_words_new]
        obj_words=[word2index[m] for m in obj.split()]

        triples_words_id.append([sub_words,pre_words,obj_words])
        if type==1:
            triples_id.append([-1,int(sp[1]),entity2id[obj]])
        elif type==2:
            triples_id.append([entity2id[sub], int(sp[1]), -1])
        else:
            triples_id.append([-1,int(sp[1]),-1])
        triples_len.append([len(sub_words),len(pre_words),len(obj_words)])

    return sentences_words_id, triples_words_id, triples_id, triples_len


sen_uk,tri_words_uk,tri_uk,tri_len_uk=process(file_uk,1)
sen_ku,tri_words_ku,tri_ku,tri_len_ku=process(file_ku,2)
sen_uu,tri_words_uu,tri_uu,tri_len_uu=process(file_uu,3)
# 1762 1323 79
print(len(sen_uk),len(sen_ku),len(sen_uu))
# 3164
print(count)
dict2pickle={
            'dict':{"emb_matrix":all_data['dict']['emb_matrix'],
                    'word2index':all_data['dict']['word2index'],
                    'index2word':all_data['dict']['index2word']},
            'kb':{
                'entity2id':all_data['kb']['entity2id'],
                'id2entity':all_data['kb']['id2entity'],
                'entity_id2words_id':all_data['kb']['entity_id2words_id'],
                'relation2id':all_data['kb']['relation2id'],
                'id2relation':all_data['kb']['id2relation'],
                'relation_id2words_id':all_data['kb']['relation_id2words_id']},
            'test_uk':{
                'sen_id':sen_uk,
                'triple_words_id':tri_words_uk,
                'triple_id':tri_uk,
                'triple_len':tri_len_uk},
            'test_ku':{
                'sen_id':sen_ku,
                'triple_words_id':tri_words_ku,
                'triple_id':tri_ku,
                'triple_len':tri_len_ku},
            'test_uu':{
                'sen_id':sen_uu,
                'triple_words_id':tri_words_uu,
                'triple_id':tri_uu,
                'triple_len':tri_len_uu}
            }
pickle.dump(dict2pickle,open('all_data_for_new.pkl','wb'))

# 最长：304
# 平均：34.14886219974716
print(max_len)
print(total_len/count)