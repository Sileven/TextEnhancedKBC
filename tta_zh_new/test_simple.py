"""实体预测任务"""
import argparse
import time
from tqdm import tqdm
import torch
import torch.utils.data
import sys
import json
sys.path.append('../')
import dataset.Constants as Constants
from dataset.dataset import KBDataset
from trans_text import Trans
#trans2
import numpy as np
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(model,data,device,opt):
    ''' Start training '''
    global max_entity_len
    entity_nums=len(data['kb']['id2entity'])
    #relation_nums=len(data['kb']['id2relation'])
    repl_entity=list(range(entity_nums))
    repl_entity_words=[entity2words[m] for m in repl_entity]
    repl_entity_len=[len(m) for m in repl_entity_words]

    with torch.no_grad():
        for line in open('sample.txt','r',encoding='utf-8'):
            j=json.loads(line.strip())
            words=j['word']
            print(''.join(words))
            words_id=[data['dict']['word2index'][w] for w in words]

            sub_words=j['spo_list'][0]['subject']
            pre_words=j['spo_list'][0]['predicate']
            obj_words=j['spo_list'][0]['object']

            print(''.join(sub_words),''.join(pre_words),''.join(obj_words))

            sub_words_id=[data['dict']['word2index'][w] for w in sub_words]
            pre_words_id=[data['dict']['word2index'][w] for w in pre_words]
            obj_words_id=[data['dict']['word2index'][w] for w in obj_words]

            sub_id=data['kb']['entity2id'][''.join(sub_words)]
            pre_id=data['kb']['relation2id'][''.join(pre_words)]
            obj_id=data['kb']['entity2id'][''.join(obj_words)]

            words_len=len(words_id)

            words=torch.LongTensor([words_id])
            words_len=torch.LongTensor([words_len])
            words=words.to(device)
            words_len=words_len.to(device)

            start=0
            f=False

            repl_head_result=[]
            repl_tail_result=[]
            hr=sub_words_id+[Constants.ER]+pre_words_id+[Constants.RE]
            rt=[Constants.ER]+pre_words_id+[Constants.RE]+obj_words_id
            while start<entity_nums:
                end=min(start+opt.test_batch_size,entity_nums)

                # 替换头实体
                triples_id_head=[]
                triples_len_head=[]
                triples_sen_head=[]
                max_head_len=0
                for repl_id,l,repl_words in zip(repl_entity[start:end],repl_entity_len[start:end],repl_entity_words[start:end]):
                    triples_id_head.append([repl_id,pre_id,obj_id])
                    triples_len_head.append([l,len(pre_words),len(obj_words)])
                    triples_sen_head.append(repl_words+rt)
                    max_head_len=max(len(triples_sen_head[-1]),max_head_len)
                triples_sen_head_padded=[m+[Constants.PAD]*(max_head_len-len(m)) for m in triples_sen_head]

                triples_id_head=torch.LongTensor(triples_id_head).to(device)
                triples_sen_head_padded=torch.LongTensor(triples_sen_head_padded).to(device)
                triples_len_head=torch.LongTensor(triples_len_head).to(device)

                if not f:
                    score_head, enc_text=model(words,words_len,triples_id_head,triples_sen_head_padded,triples_len_head,None,'pos',True)
                    f=True
                else:
                    score_head, enc_text=model(words,words_len,triples_id_head,triples_sen_head_padded,triples_len_head,enc_text,'neg',True)

                # 替换尾实体
                triples_id_tail=[]
                triples_len_tail=[]
                triples_sen_tail=[]
                max_tail_len=0
                for repl_id,l,repl_words in zip(repl_entity[start:end],repl_entity_len[start:end],repl_entity_words[start:end]):
                    triples_id_tail.append([sub_id,pre_id,repl_id])
                    triples_len_tail.append([len(sub_words),len(pre_words),l])
                    triples_sen_tail.append(hr+repl_words)
                    max_tail_len=max(len(triples_sen_tail[-1]),max_tail_len)
                triples_sen_tail_padded=[m+[Constants.PAD]*(max_tail_len-len(m)) for m in triples_sen_tail]

                triples_id_tail = torch.LongTensor(triples_id_tail).to(device)
                triples_sen_tail_padded = torch.LongTensor(triples_sen_tail_padded).to(device)
                triples_len_tail = torch.LongTensor(triples_len_tail).to(device)

                score_tail, enc_text=model(words, words_len, triples_id_tail, triples_sen_tail_padded,triples_len_tail,enc_text,'neg',True)

                repl_head_result.append(score_head)
                repl_tail_result.append(score_tail)

                start=end

            score_head = torch.cat(repl_head_result, 0)
            score_tail = torch.cat(repl_tail_result, 0)

            _, index_head = torch.sort(score_head)
            _, index_tail = torch.sort(score_tail)

            triple=[sub_id,pre_id,obj_id]
            rank_head = np.where(index_head.cpu().numpy() == triple[0])[0][0]
            rank_tail = np.where(index_tail.cpu().numpy() == triple[2])[0][0]

            for hh in range(0,rank_head):
                print(data['kb']['id2entity'][index_head.cpu().numpy()[hh]],end=' ')
            print()
            for hh in range(0,rank_tail):
                print(data['kb']['id2entity'][index_tail.cpu().numpy()[hh]],end=' ')

            print(rank_head,rank_tail)


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-log', default=None)
    parser.add_argument('-no_cuda', action='store_true',default=False)
    parser.add_argument('-test_batch_size',default=300)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    '''========= Loading Dataset ========='''
    data = pickle.load(open('../dataset/all_data_new_correct.pkl','rb'))
    emb_matrix=data['dict']['emb_matrix']

    global entity2words
    #global relation2words

    entity2words = data['kb']['entity_id2words_id']
    #relation2words = data['kb']['relation_id2words_id']

    opt.vocab_size = emb_matrix.shape[0]

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Trans(
        emb_matrix,
        len(data['kb']['id2entity']),
        len(data['kb']['id2relation']),
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        dropout=opt.dropout).to(device)

    #path='checkpoint/2019-06-06_17:36:38'
    path='checkpoint/2019-06-30_11:42:55'
    model_path=path+'/model300'
    print('load model from',model_path)
    checkpoint=torch.load(model_path)
    transformer.load_state_dict(checkpoint['model'])
    #opt=checkpoint['settings']
    print('trian opt',checkpoint['settings'])
    print('test opt',opt)
    transformer.eval()
    test(transformer,data, device,opt)



if __name__ == '__main__':
    main()

