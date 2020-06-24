# -*- coding: utf-8 -*-
"""
@Author: Eleven
@Date  : 2019/3/21 11:12 AM
把已知实体kb分为train 265673和test 52116
"""
import json
file_k=open('split/known.json','r',encoding='utf-8')

# 记录每个关系对应的三元组数
rel2count={}
# 记录每个关系对应的三元组
rel2triples={}

entity2count={}
"""统计"""
for line in file_k:
    j = json.loads(line)
    spo=j['spo']
    sub=''.join(spo['subject'])
    pre=''.join(spo['predicate'])
    obj=''.join(spo['object'])

    # 关系 和对应的 频次
    if pre in rel2count:
        rel2count[pre] += 1
    else:
        rel2count[pre] = 1

    # 实体和实体对应的频次
    if sub in entity2count:
        entity2count[sub] += 1
    else:
        entity2count[sub] = 1

    if obj in entity2count:
        entity2count[obj] += 1
    else:
        entity2count[obj] = 1

    # 关系 和对应的 三元组set
    if pre not in rel2triples:
        rel2triples[pre] = set()
    rel2triples[pre].add((sub, pre, obj))
file_k.close()
count=0
for r,t in rel2triples.items():
    count+=len(t)
"""count是所有的关系对应的三元组数之和 247024个"""
print(count)

test_rel2triples={}
entity_tmp=set()
count_all=0
rel2count_list=sorted(rel2count.items(),key=lambda item:item[1],reverse=True)
"""分割train和test"""
for rel,count in rel2count_list:
    num=int(count/10)
    n=0
    test_rel2triples[rel]=set()

    triples_sorted=sorted(rel2triples[rel],key=lambda a:entity2count[a[0]],reverse=True)
    for triples in triples_sorted:
        if triples[0] not in entity_tmp and entity2count[triples[0]]>1:
            test_rel2triples[rel].add(triples)
            entity_tmp.add(triples[0])
            n+=1
            count_all+=1
        if n>=num:
            break
    # num是rel要分出来的test的数量
    print(rel,num,n)

ccount=0
for tr,tt in test_rel2triples.items():
    ccount+=len(tt)
'''test集中的三元组个数 31000'''
print(count_all,ccount)
file_k=open('split/known.json','r',encoding='utf-8')
file_k_train=open('split/known_train.json','w',encoding='utf-8')
file_k_test=open('split/known_test.json','w',encoding='utf-8')

count1=0
count2=0
for line in file_k:
    j = json.loads(line)
    spo=j['spo']
    sub=''.join(spo['subject'])
    pre=''.join(spo['predicate'])
    obj=''.join(spo['object'])
    if (sub,pre,obj) in test_rel2triples[pre]:
        file_k_test.write(json.dumps(j,ensure_ascii=False)+'\n')
        count1+=1
    else:
        file_k_train.write(json.dumps(j,ensure_ascii=False)+'\n')
        count2+=1
'''实际分出来的train和test数目，因为一个三元组可能对应多个句子，每行一个句子一个三元组'''
'''52013 265789'''
print(count1,count2)
file_k_test.close()
file_k_train.close()