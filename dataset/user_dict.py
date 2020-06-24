"""
通过raw数据中的“人物”类别的实体获得语料中的人名词及其频次，方便后续分词。

"""
from pyhanlp import *
import pickle
import json
import re
import jieba
file1=open('raw/train_data.json','r',encoding='utf-8')
file2=open('raw/dev_data.json','r',encoding='utf-8')

entity2count={}


def count(file):
    for line in file:
        j = json.loads(line)
        spo_list = j['spo_list']
        for spo in spo_list:
            if spo['subject_type']=='人物':
                sp=re.split('[・· ]',spo['subject'])
                for ss in sp:
                    if ss in entity2count:
                        entity2count[ss]+=1
                    else:
                        entity2count[ss] = 1
            if spo['object_type']=='人物':
                sp=re.split('[・· ]',spo['object'])
                for ss in sp:
                    if ss in entity2count:
                        entity2count[ss]+=1
                    else:
                        entity2count[ss] = 1

count(file1)
count(file2)
sorted_en2count=sorted(entity2count.items(),key=lambda item:item[1],reverse=True)

file3=open('dict/user_dict.txt','w',encoding='utf-8')
file4=open('dict/user_dict_hanlp.txt','w',encoding='utf-8')
for en,c in sorted_en2count:
    en=re.sub("\s+","",en.strip())
    #print(en)
    if (not en.encode('utf-8').isdigit()) and (not en.encode('utf-8').isalpha()) and (not en.encode('utf-8').isalnum() and len(en)<5):
        file3.write(en+' '+str(c)+' nr'+'\n')
        file4.write(en+' nr '+str(c)+'\n')