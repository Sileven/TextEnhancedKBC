"""对语料重新分词，处理实体不在分词结果中的情况，输出到新文件"""
from pyhanlp import *
import pickle
import json
import jieba
import copy
import re
jieba.load_userdict('dict/user_dict.txt')
file_train_json=open('raw/train_data.json','r',encoding='utf-8')
file_train_txt=open('my_seg/train_data.txt','w',encoding='utf-8')

file_dev_json=open('raw/dev_data.json','r',encoding='utf-8')
file_dev_txt=open('my_seg/dev_data.txt','w',encoding='utf-8')

#hanlp分词并检查是否在预训练词向量词表中
# word_pre=pickle.load(open('../word_pre.pkl','rb'))
word_all=set()
#word_not=set()
entity=set()
entity_words=set()
relation=set()
count=0
count_bad=0
bad_relation={}
bad_word={}
bbb=0

def sub_obj(words,spo,target,flag):
    """不对实体分词"""
    global count_bad
    global count
    if spo[target] in ['学文','任广']:
        return [],flag
    # if len(spo[target])<5:
    #     # 实体直接就在文本分词结果中出现
    #     if spo[target] in words:
    #         entity_words.add(spo[target])
    #         spo[target]=[spo[target]]
    #         return words,flag
    #
    #     # 实体是文本分词结果1-N个词的一部分
    #     for size in range(1,4):
    #         i=0
    #         words_new = []
    #         while i<len(words):
    #             w=''.join(words[i:min(len(words),i+size)])
    #             if spo[target] in w:
    #                 index = w.index(spo[target])
    #                 if index != 0:
    #                     words_new.append(w[:index])
    #                 words_new.append(spo[target])
    #                 if index + len(spo[target]) != len(w):
    #                     words_new.append(w[index + len(spo[target]):])
    #                 i += size
    #                 # print(count,words_new)
    #             else:
    #                 words_new.append(words[i])
    #                 if size>1 and i == len(words) - size:
    #                     words_new.extend(words[i+1:i+size])
    #                 i += 1
    #                 # print(count,words_new)
    #
    #         if spo[target] in words_new:
    #             words=words_new
    #             entity_words.add(spo[target])
    #             spo[target]=[spo[target]]
    #             return words,flag

    # 文本分词结果的1-n个词是实体的一部分，相当于用文本内的词对实体分词，控制实体的分词每词长大于1

    """对实体分词"""
    target_words = re.sub('([a-zA-Z])\s+([a-zA-Z])', '\g<1>\\\P\g<2>', spo[target])
    target_words = re.sub('\s+', '', target_words)
    target_words = target_words.replace('\P', ' ')
    seg=jieba.lcut(target_words)
    seg = [m.lower() for m in seg if m.strip() != '']
    # seg = HanLP.segment(spo[target])
    # seg = [m.word.strip().lower() for m in seg if m.word.strip()!='']
    # if count == 108256:
    #     print(seg)
    # 先拆实体词

    # seg_new=[]
    # for s in seg:
    #     if s not in words:
    #         i=0
    #         ff=False
    #         # 从头遍历text
    #         while i<len(words):
    #             if s.startswith(words[i]):
    #                 tmp=[words[i]]
    #                 j=i+1
    #                 while j<len(words):
    #                     tmp.append(words[j])
    #                     if ''.join(tmp)==s:
    #                         seg_new.extend(tmp)
    #                         ff=True
    #                         break
    #                     if s.startswith(''.join(tmp)):
    #                         j+=1
    #                     else:
    #                         break
    #                 if ff:
    #                     break
    #                 else:
    #                     i+=1
    #             else:
    #                 i+=1
    #
    #         if not ff:
    #             seg_new.append(s)
    #     else:
    #         seg_new.append(s)
    #     # if count == 77:
    #     #     print(seg_new)
    #     #     print(words)
    # seg=seg_new

    # 再合并或拆分text词
    for s in seg:
        if s not in words:
            # 实体是文本分词结果1-N个词的一部分
            for size in range(1, 5):
                i = 0
                words_new = []

                while i < len(words):
                    w = ''.join(words[i:min(i + size,len(words))])
                    if s in w:
                        index = w.index(s)
                        if index != 0:
                            words_new.append(w[:index])
                        words_new.append(s)
                        if index + len(s) != len(w):
                            words_new.append(w[index + len(s):])
                        i += size
                    # if w.startswith(s):
                    #     words_new.append(s)
                    #     words_new.append(w[len(s):])
                    #     i+=size
                    # elif w.endswith(s):
                    #     words_new.append(w[:-len(s)])
                    #     words_new.append(s)
                    #     i+=size
                    else:
                        words_new.append(words[i])
                        # if size > 1 and i == len(words) - size:
                        #     words_new.extend(words[i + 1:i + size])
                        i += 1
                # if count==77:
                #     print(words_new)
                if s in words_new:
                    words = words_new
                    break

            if s not in words:
                if flag:
                    count_bad += 1
                    flag = False
                if spo['predicate'] in bad_relation:
                    bad_relation[spo['predicate']] += 1
                else:
                    bad_relation[spo['predicate']] = 1
                print(count, words_new,seg, s)
                if s in bad_word:
                    bad_word[s]+=1
                else:
                    bad_word[s]=1
                return [],flag

    spo[target]=seg
    # for s in seg:
    #     entity_words.add(s)
    return words,flag
def seg_file(file1,file2):
    global count
    global count_bad
    global bbb
    for line in file1:
        #try:
        j = json.loads(line)
        # except:
        #     print(count)
        #     exit(0)
        text = j['text']
        spo_list = j['spo_list']
        # 去除空格
        # text = re.sub('([a-zA-Z])\s+([a-zA-Z])', '\g<1>\\\P\g<2>', text)
        # text = re.sub('\s+', '', text)
        # text = text.replace('\P', ' ')
        text_seg=jieba.lcut(text)
        words=[m.strip().lower() for m in text_seg if m.strip()!='']
        # text_seg=HanLP.segment(text)
        # words=[m.word.strip().lower() for m in text_seg if m.word.strip()!='']
        j_out={}
        flag=True
        new_list=[]

        # 删除多余spo
        spo_list2=[spo_list[0]]
        for spo in spo_list:
            yes = False
            for j in range(len(spo_list2)):
                if spo['predicate'] == spo_list2[j]['predicate']:
                    if spo['object'] == spo_list2[j]['object']:
                        if spo_list2[j]['subject'] in spo['subject']:
                            spo_list2[j] = spo
                            yes = False
                            break
                        elif spo['subject'] in spo_list2[j]['subject']:
                            yes = False
                            break
                    elif spo['subject'] == spo_list2[j]['subject']:
                        if spo_list2[j]['object'] in spo['object']:
                            spo_list2[j] = spo
                            yes = False
                            break
                        elif spo['object'] in spo_list2[j]['object']:
                            yes = False
                            break
                    else:
                        yes = True
                else:
                    yes = True
            if yes:
                spo_list2.append(spo)
        spo_list=spo_list2
        # if count==59648:
        #
        #     print(spo_list)

        spo_sorted_list=sorted(spo_list,key=lambda item:len(item['object']))
        for spo in spo_sorted_list:
            entity.add(spo['subject'])
            entity.add(spo['object'])
            relation.add(spo['predicate'])
            # if count==108256:
            #     print(spo)
            #     print(words)
            words_new,flag=sub_obj(words,spo,"subject",flag)
            if words_new:
                words_new,flag=sub_obj(words_new,spo,"object",flag)
                if words_new:
                    words=words_new
                    new_list.append(spo)
            # if count==108256:
            #     print(spo)
            #     print(words)
            #     print()

        if not new_list:
            count+=1
            continue
        flag_error=False
        for spo in new_list:
            for sub in spo['subject']:
                if sub not in words:
                    # print(new_list)
                    # print(count, spo['subject'],sub, words)
                    # print()
                    flag_error=True
            for obj in spo['object']:
                if obj not in words:
                    # print(new_list)
                    # print(count,spo['object'],obj,words)
                    # print()
                    flag_error=True
        if flag_error:
            bbb+=1
        # for m in words:
        #     word_all.add(m)

        j_out['word']=words
        j_out['spo_list']=new_list
        if not flag_error:
            file2.write(json.dumps(j_out,ensure_ascii=False)+'\n')
            for m in words:
                word_all.add(m)
            for spo in new_list:
                for sub in spo['subject']:
                    entity_words.add(sub)
                for obj in spo['object']:
                    entity_words.add(obj)
        count+=1
        if count%10000==0:
            print(count,'done')
            print(len(word_all))


seg_file(file_train_json,file_train_txt)
seg_file(file_dev_json,file_dev_txt)

error=0
for en_w in entity_words:
    if en_w not in word_all:
        print(en_w)
        error+=1

# 292927 0 169351
print(len(word_all),error,len(entity_words))
# 194748 0
print(count,count_bad)

bad_relation_sorted=sorted(bad_relation.items(),key=lambda item:item[1],reverse=True)
print(bad_relation_sorted)
bad_word_sorted=sorted(bad_word.items(),key=lambda item:item[1],reverse=True)
print(bad_word_sorted)

# 1019
print(bbb)
# 231799 49
print(len(entity),len(relation))

error=0
for r in relation:
    if r not in word_all:
        rr=jieba.lcut(r)
        for rrr in rr:
            if rrr not in word_all:
                print(rrr)
                error+=1
print(error)




