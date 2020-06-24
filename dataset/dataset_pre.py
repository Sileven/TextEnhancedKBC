# -*- coding: utf-8 -*-
"""
@Author: Eleven
@Date  : 2019/3/19 1:41 PM
把分好词的train_data和dev_data分成已知实体kb和未知实体kb（uu,uk,ku）
"""
import json
import copy
file_train=open('my_seg/train_data.txt','r',encoding='utf-8')
file_dev=open('my_seg/dev_data.txt','r',encoding='utf-8')
file_schema=open('raw/all_50_schemas','r',encoding='utf-8')
# 记录每个关系对应的三元组数
rel2count={}
# 记录每个关系对应的头实体
rel2heads={}
rel2tails={}
# 记录每个关系对应的三元组
rel2triples={}
# 记录每个关系对应的头尾实体和三元组
rel2head_triple={}
rel2head_tails={}
rel2tail_triple={}
rel2tail_heads={}
entity2count={}
count_all=0

def count_dataset(file):
    entity=set()
    triple=set()
    predicate=set()
    object=set()
    subject=set()
    for line in file:
        j=json.loads(line)
        for spo in j['spo_list']:
            sub=''.join(spo['subject'])
            pre=''.join(spo['predicate'])
            obj=''.join(spo['object'])

            predicate.add(pre)
            object.add(obj)
            subject.add(sub)

            triple.add((sub,pre,obj))

            entity.add(obj)
            entity.add(sub)

            # 关系 和对应的 频次
            if pre in rel2count:
                rel2count[pre]+=1
            else:
                rel2count[pre]=1

            if sub in entity2count:
                entity2count[sub]+=1
            else:
                entity2count[sub]=1

            if obj in entity2count:
                entity2count[obj]+=1
            else:
                entity2count[obj]=1

            # 关系 和对应的 三元组
            if pre not in rel2triples:
                rel2triples[pre] = set()
            rel2triples[pre].add((sub,pre,obj))

            if pre not in rel2heads:
                # 关系和对应的 头尾实体 和对应的 频次
                rel2heads[pre] = {}
                rel2tails[pre] = {}
                # 关系和对应的 头尾实体 和对应的 三元组
                rel2head_triple[pre]={}
                rel2tail_triple[pre]={}
                # 关系和对应的 头尾实体 和对应的 尾头实体集合
                rel2tail_heads[pre]={}
                rel2head_tails[pre]={}

            # 关系 和对应的 头实体 和对应的 *
            if sub in rel2heads[pre]:
                rel2heads[pre][sub]+=1
            else:
                rel2heads[pre][sub]=1
                rel2head_triple[pre][sub]=set()
                rel2head_tails[pre][sub]=set()
            rel2head_triple[pre][sub].add((sub, pre, obj))
            rel2head_tails[pre][sub].add(obj)

            # 关系 和对应的 尾实体 和对应的 *
            if obj in rel2tails[pre]:
                rel2tails[pre][obj] += 1
            else:
                rel2tails[pre][obj] =1
                rel2tail_triple[pre][obj] = set()
                rel2tail_heads[pre][obj]=set()
            rel2tail_triple[pre][obj].add((sub, pre, obj))
            rel2tail_heads[pre][obj].add(sub)

    return predicate,object,subject,triple,entity


# 统计训练集验证集中数据
pre_train,obj_train,sub_train,triple_train,entity_train=count_dataset(file_train)
pre_dev,obj_dev,sub_dev,triple_dev,entity_dev=count_dataset(file_dev)
file_train.close()
file_dev.close()
print('count dataset done')

# print(len(entity2count))
# ccc=0
# for e,c in entity2count.items():
#     if c<=5:
#         ccc+=1
# print(ccc)

# count_all=0
# count_r=0
# for r,c in rel2count.items():
#     count_all+=c
#     count_r+=len(rel2triples[r])
# print(count_all,count_r)
# exit(0)

# 对rel2heads各个rel对应的heads按出现次数排序
for rel,heads in rel2heads.items():
    heads_list=sorted(heads.items(),key= lambda item:item[1])
    rel2heads[rel]=heads_list
# 对rel2tails各个rel对应的tails按出现次数排序
for rel,tails in rel2tails.items():
    tails_list=sorted(tails.items(),key= lambda item:item[1])
    rel2tails[rel]=tails_list

# 对rel2count 按count排序,从小到大
rel2count_list=sorted(rel2count.items(),key=lambda item:item[1])

# 打印所有的关系 在数据集中出现的次数和对饮的三元组（无重复）个数
for rel,count in rel2count_list:
    print(rel,count,len(rel2triples[rel]))

# 记录全部的关系
all_schemas=set()
for line in file_schema:
    j=json.loads(line)
    all_schemas.add(j['predicate'])

print(len(sub_train),len(pre_train),len(obj_train),len(triple_train),len(entity_train))
print(len(sub_dev),len(pre_dev),len(obj_dev),len(triple_dev),len(entity_dev))
print(len(all_schemas))
print(len(sub_train-sub_dev),len(obj_train-obj_dev),len(triple_train-triple_dev),len(entity_train-entity_dev))
print(len(sub_dev-sub_train),len(obj_dev-obj_train),len(triple_dev-triple_train),len(entity_dev-entity_train))
print(len(sub_train&sub_dev),len(obj_train&obj_dev),len(triple_train&triple_dev),len(entity_dev&entity_train))

u_rel2triples={}
u_rel2count={}
u_entity=set()


def add2unknown(relation,relation_triple,new_entity,nn):
    if rel in u_rel2triples:
        u_rel2triples[relation] = u_rel2triples[relation] | relation_triple[relation][new_entity]
        u_rel2count[relation] += len(relation_triple[relation][new_entity])
    else:
        u_rel2triples[relation] = relation_triple[relation][new_entity]
        u_rel2count[relation] = len(relation_triple[relation][new_entity])
    nn += len(relation_triple[relation][new_entity])
    return nn


# 区分已知实体集和未知实体集
for rel,rel_count in rel2count_list:
    nums=int(rel_count/20)
    n=0
    n_real=0
    # 把包含未知实体集中的实体的三元组都抽出来
    u_entity_copy=copy.copy(u_entity)
    for ue in u_entity_copy:
        if ue in rel2head_triple[rel]:
            n=add2unknown(rel,rel2head_triple,ue,n)

        if ue in rel2tail_triple[rel]:
            n=add2unknown(rel,rel2tail_triple,ue,n)

    if rel in u_rel2triples:
        n_real=len(u_rel2triples[rel])
    # 选头实体和尾实体交替作为未知实体
    flag=0
    index_head=0
    index_tail=0
    while n_real<nums and (index_head<len(rel2heads[rel]) or index_tail<len(rel2tails[rel])):
        # 随机取一个头实体作为新的未知实体
        if flag==0:
            # 头实体列表遍历完了
            if index_head>=len(rel2heads[rel]):
                flag=1
                continue

            head_now,head_count=rel2heads[rel][index_head]
            if head_now in u_entity:
                index_head+=1
                continue

            n=add2unknown(rel,rel2head_triple,head_now,n)
            u_entity.add(head_now)

            index_head+=1
            if index_tail < len(rel2tails[rel]):
                flag=1
        # 随机取一个尾实体作为新的未知实体
        else:
            # 尾实体列表遍历完了
            if index_tail>=len(rel2tails[rel]):
                flag=0
                continue
            tail_now,tail_count=rel2tails[rel][index_tail]
            if tail_now in u_entity:
                index_tail+=1
                continue

            n=add2unknown(rel,rel2tail_triple,tail_now,n)
            u_entity.add(tail_now)

            index_tail+=1
            if index_head<len(rel2heads[rel]):
                flag=0

        if rel in u_rel2triples:
            n_real = len(u_rel2triples[rel])
    # nums是理论上rel要分出来多少三元组作为未知实体三元组，n为实际分出来多少三元组（可能有重复）。u_rel2triples是每个rel分出来的未知三元组的个数（无重复）
    print(rel,'理论:',nums,'实际:',n,'无重复:',len(u_rel2triples[rel]),'当前未知实体个数:',len(u_entity))

# 二次筛选
print(len(u_entity))
for rel,rel_count in rel2count_list:
    print(rel,'频次:',rel_count,'无重复频次:',len(rel2triples[rel]),'第一次:',len(u_rel2triples[rel]),end=' ')
    n=0
    for ue in u_entity:
        if ue in rel2head_triple[rel]:
            n=add2unknown(rel,rel2head_triple,ue,n)

        if ue in rel2tail_triple[rel]:
            n=add2unknown(rel,rel2tail_triple,ue,n)
    print('第二次:',len(u_rel2triples[rel]))
# 统计未知实体集的占比
de_all=0
ed_all=0
dd_all=0
for rel,triples in u_rel2triples.items():
    de=0
    ed=0
    dd=0
    for triple in triples:
        if triple[0] in u_entity and triple[2] in u_entity:
            dd+=1
        if triple[0] not in u_entity and triple[2] in u_entity:
            ed+=1
        if triple[0] in u_entity and triple[2] not in u_entity:
            de+=1
    print(rel,de,ed,dd)
    dd_all+=dd
    de_all+=de
    ed_all+=ed
# 9093 23950 3999
print(de_all,ed_all,dd_all)

# 把数据集拆分
file_k=open('split/known.json','w',encoding='utf-8')
file_uk=open('split/unknown_uk.json','w',encoding='utf-8')
file_ku=open('split/unknown_ku.json','w',encoding='utf-8')
file_uu=open('split/unknown_uu.json','w',encoding='utf-8')

file_train=open('my_seg/train_data.txt','r',encoding='utf-8')
file_dev=open('my_seg/dev_data.txt','r',encoding='utf-8')

def split_dataset(file):
    for line in file:
        j=json.loads(line)
        json_tmp={}
        json_tmp['word']=j['word']
        for spo in j['spo_list']:
            sub=''.join(spo['subject'])
            pre=''.join(spo['predicate'])
            obj=''.join(spo['object'])

            json_tmp['spo'] = spo

            if (sub,pre,obj) in u_rel2triples[pre]:
                if sub in u_entity and obj in u_entity:
                    file_uu.write(json.dumps(json_tmp,ensure_ascii=False)+'\n')
                elif sub not in u_entity and obj in u_entity:
                    file_ku.write(json.dumps(json_tmp,ensure_ascii=False)+'\n')
                elif sub in u_entity and obj not in u_entity:
                    file_uk.write(json.dumps(json_tmp,ensure_ascii=False)+'\n')
            else:
                file_k.write(json.dumps(json_tmp,ensure_ascii=False)+'\n')


split_dataset(file_train)
split_dataset(file_dev)
file_train.close()
file_dev.close()
