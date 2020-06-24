"""关系预测任务"""
import argparse
import time
from tqdm import tqdm
import torch
import torch.utils.data
import sys
sys.path.append('../')
import dataset.Constants as Constants
from dataset.dataset import KBDataset
from trans2 import Trans
import numpy as np
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test(model,data,test_data,device,opt):
    ''' Start training '''
    relation_nums=len(data['kb']['id2relation'])
    repl_relation=list(range(relation_nums))
    repl_relation_words=[relation2words[m] for m in repl_relation]
    repl_relation_len=[len(m) for m in repl_relation_words]

    mean_rank = 0
    hits10 = 0
    hits3=0
    hits1=0

    with torch.no_grad():
        start_time = time.time()
        count=0
        for batch in tqdm(test_data, mininterval=2, desc='  - (Testing)   ', leave=False):
            words, words_len, triples_id, triples_words, triples_len= batch
            words=words.to(device)
            words_len=words_len.to(device)

            hp=triples_words[0][0]+[Constants.ER]
            pt=[Constants.RE]+triples_words[0][2]

            # 替换关系
            triples_id_rel=[]
            triples_len_rel=[]
            triples_sen_rel=[]
            max_rel_len=0
            for repl_id,l,repl_words in zip(repl_relation,repl_relation_len,repl_relation_words):
                triples_id_rel.append([triples_id[0][0],repl_id,triples_id[0][2]])
                triples_len_rel.append([triples_len[0][0],l,triples_len[0][2]])
                triples_sen_rel.append(hp+repl_words+pt)
                max_rel_len=max(len(triples_sen_rel[-1]),max_rel_len)
            triples_sen_rel_padded=[m+[Constants.PAD]*(max_rel_len-len(m)) for m in triples_sen_rel]

            triples_id_rel=torch.LongTensor(triples_id_rel).to(device)
            triples_sen_rel_padded=torch.LongTensor(triples_sen_rel_padded).to(device)
            triples_len_rel=torch.LongTensor(triples_len_rel).to(device)

            score_rel, _ = model(words,words_len,triples_id_rel,triples_sen_rel_padded,triples_len_rel,None,'pos',True)

            del triples_id_rel
            del triples_sen_rel
            del triples_len_rel

            _, index_rel = torch.sort(score_rel)

            del score_rel
            del words
            del words_len

            triple=triples_id[0]
            rank_rel = np.where(index_rel.cpu().numpy() == triple[1])[0][0]

            del index_rel

            mean_rank += (rank_rel + 1)

            if rank_rel + 1 <= 10:
                hits10 += 1
                if rank_rel+1<=3:
                    hits3+=1
                    if rank_rel==0:
                        hits1+=1

            count+=1
            if count%100==0:
                print('test: {nums:d}, cost: {cost:3.3f} min'.format(nums=count,cost=(time.time() - start_time) / 60))
                print('mean_rank:{:.1f}, hits@10:{:.4f}, hits@3:{:.4f}, hits@1:{:.4f}'.format(mean_rank/count, hits10/count, hits3/count, hits1/count))
                start_time=time.time()

            torch.cuda.empty_cache()
        print('mean_rank:{:.1f}, hits@10:{:.4f}, hits@3:{:.4f}, hits@1:{:.4f}'.format(mean_rank / count, hits10 / count,
                                                                                      hits3 / count, hits1 / count))

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

    global relation2words
    relation2words=data['kb']['relation_id2words_id']

    opt.vocab_size = emb_matrix.shape[0]
    test_data = prepare_dataloaders(data)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Trans(
        emb_matrix,
        len(data['kb']['id2entity']),
        len(data['kb']['id2relation']),
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        dropout=opt.dropout).to(device)

    path='checkpoint/2019-07-08_01:18:26'
    model_path=path+'/model300'
    print('load model from',model_path)
    checkpoint=torch.load(model_path)
    transformer.load_state_dict(checkpoint['model'])
    #opt=checkpoint['settings']
    print('trian opt',checkpoint['settings'])
    print('test opt',opt)
    transformer.eval()
    test(transformer,data,test_data, device,opt)


def prepare_dataloaders(data):
    # ========= Preparing DataLoader =========#
    valid_loader = torch.utils.data.DataLoader(
        KBDataset(
            words=data['test']['sen_id'],
            triples=data['test']['triple_id'],
            triples_words=data['test']['triple_words_id'],
            triple_len=data['test']['triple_len'],
            types=data['test']['type_id'],
            ),
        num_workers=1,
        batch_size=1,
        collate_fn= paired_collate_fn,
        shuffle=False)
    return valid_loader


def paired_collate_fn(insts):
    words,triples_id,triples_words,triples_len,types=list(zip(*insts))

    # 文本pad
    words_len = [len(sen) for sen in words]
    max_words_len = max(words_len)
    words_padded = np.array([sen + [Constants.PAD] * (max_words_len - len(sen)) for sen in words])

    batch_words=torch.LongTensor(words_padded)
    batch_words_len=torch.LongTensor(words_len)

    return batch_words, batch_words_len, \
           triples_id, triples_words, triples_len


if __name__ == '__main__':
    main()

