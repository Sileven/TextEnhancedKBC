"""统计FB20K中多少实体是已知，多少是未知"""
import os
import re
root='/home/sjq/data/FB15K/FB20K-new'

new_entity_name_path=os.path.join(root,'entityName.txt')
new_triple_path=os.path.join(root,'triple.txt')

new_entity2name={}
new_names=set()
new_repeat=0
with open(new_entity_name_path,'r',encoding='utf-8') as f:
    for line in f:
        sp = line.strip().split('\t')
        name = re.sub('"(.+)"@en.*','\g<1>',sp[1]).lower()
        name = re.sub('\\\\','',name)
        name = re.sub('"','',name)
        print(name)
        new_entity2name[sp[0]] = name
        if name in new_names:
            new_repeat+=1
        else:
            new_names.add(name)

old_entity2name={}
old_names=set()
old_repeat=0
file_fb15k_entity2name = open('/home/sjq/data/FB15K/description/mid2name.txt', 'r', encoding='utf-8')
for line in file_fb15k_entity2name:
    sp = line.strip().split('\t')
    name=re.sub("_+",' ',sp[1]).lower()
    old_entity2name[sp[0]]=name
    if name in old_names:
        old_repeat+=1
    else:
        old_names.add(name)
file_fb15k_entity2name.close()
print('load entity names done')

head_known=0
head_unknown=0
head_error=0
tail_known=0
tail_unknown=0
tail_error=0

k_k=0
k_u=0
u_k=0
u_u=0
u_u_equal=0
with open(new_triple_path,'r',encoding='utf-8') as f:
    for line in f:
        sp=line.strip().split()
        flag_head='e'
        flag_tail='e'
        if sp[0] in old_entity2name:
            head_known+=1
            flag_head='k'
        elif sp[0] in new_entity2name:
            head_unknown+=1
            flag_head='u'
        else:
            head_error+=1

        if sp[1] in old_entity2name:
            tail_known+=1
            flag_tail='k'
        elif sp[1] in new_entity2name:
            tail_unknown+=1
            flag_tail='u'
        else:
            tail_error+=1

        if flag_head=='k':
            if flag_tail=='k':
                k_k+=1
            else:
                k_u+=1
        else:
            if flag_tail=='k':
                u_k+=1
            else:
                if sp[0]==sp[1]:
                    u_u_equal+=1
                u_u+=1

# 5019 14951
print(len(new_entity2name),len(old_entity2name))
# 11880 19198 0
# 19047 12031 0
print(head_known,head_unknown,head_error)
print(tail_known,tail_unknown,tail_error)
# 0 11880 19047 151 100
print(k_k,k_u,u_k,u_u,u_u_equal)
# 289 5
print(old_repeat,new_repeat)
overlap=0
for e in new_names:
    if e in old_names:
        overlap+=1
# 33
print(overlap)

