# -*- coding: utf-8 -*-
"""
@Author: Eleven
@Date  : 2019/3/20 9:53 PM
验证已知实体kb(317789)和未知实体kb的正确性，uu 11149,uk 10354,ku 23779
"""
import json
file_k=open('split/known.json','r',encoding='utf-8')
file_uk=open('split/unknown_uk.json','r',encoding='utf-8')
file_ku=open('split/unknown_ku.json','r',encoding='utf-8')
file_uu=open('split/unknown_uu.json','r',encoding='utf-8')

file_uk1=open('split/unknown_uk1.json','w',encoding='utf-8')
file_ku1=open('split/unknown_ku1.json','w',encoding='utf-8')
file_uu1=open('split/unknown_uu1.json','w',encoding='utf-8')

entity=set()

for line in file_k:
    j=json.loads(line)
    spo=j['spo']
    sub = ''.join(spo['subject'])
    obj = ''.join(spo['object'])
    entity.add(sub)
    entity.add(obj)
print('k done')
print(len(entity))
#exit(0)
error=0
count=0
for line in file_uk:
    count+=1
    j=json.loads(line)
    spo=j['spo']
    sub = ''.join(spo['subject'])
    obj = ''.join(spo['object'])
    if sub in entity:
        error+=1
        exit(1)
    if obj not in entity:
        error+=1
        file_uu1.write(json.dumps(j,ensure_ascii=False)+'\n')
        continue
    file_uk1.write(json.dumps(j,ensure_ascii=False) + '\n')
print('uk done',error)
error=0
count=0
for line in file_ku:
    count+=1
    j=json.loads(line)
    spo=j['spo']
    sub = ''.join(spo['subject'])
    obj = ''.join(spo['object'])
    if sub not in entity:
        error+=1
        file_uu1.write(json.dumps(j,ensure_ascii=False)+'\n')
        continue
    if obj in entity:
        error+=1
        exit(1)
    file_ku1.write(json.dumps(j,ensure_ascii=False) + '\n')

print('ku done',error)
error=0
count=0
for line in file_uu:
    count+=1
    j=json.loads(line)
    spo=j['spo']
    sub = ''.join(spo['subject'])
    obj = ''.join(spo['object'])
    if sub in entity:
        error+=1
        exit(1)
    if obj in entity:
        error+=1
        exit(1)
    file_uu1.write(json.dumps(j,ensure_ascii=False) + '\n')
print('uu done',error)