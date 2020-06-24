"""实体预测任务"""
import argparse
import time
from tqdm import tqdm
import torch
import torch.utils.data
import sys
sys.path.append('../')
import dataset_en.Constants as Constants
from dataset_en.dataset import KBDataset
from trans_mul_cat import Trans
#trans2
import numpy as np
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#0

def print_all_relation(mean_rank,hits10,hits3,hits1,relation2mean_rank,relation2hits10,relation2num,data,count):
    print("left:%.4f\t%.4f\t%.4f\t%.4f" % (mean_rank[0] / count, hits10[0] / count, hits3[0] / count, hits1[0] / count))
    print("right:%.4f\t%.4f\t%.4f\t%.4f\n" % (mean_rank[1] / count, hits10[1] / count, hits3[1] / count, hits1[1] / count))

    # for rel, count in relation2mean_rank.items():
    #     print("%s\t%d" % (data['kb']['id2relation'][rel], relation2num[rel]))
    #     print("all:%.2f\t%.2f" % (
    #         (relation2mean_rank[rel][0] + relation2mean_rank[rel][1]) / (2 * relation2num[rel]),
    #         (relation2hits10[rel][0] + relation2hits10[rel][1]) / (2 * relation2num[rel])))
    #     print("left:%.2f\t%.2f" % (
    #         relation2mean_rank[rel][0] / relation2num[rel], relation2hits10[rel][0] / relation2num[rel]))
    #     print("right:%.2f\t%.2f" % (
    #         relation2mean_rank[rel][1] / relation2num[rel], relation2hits10[rel][1] / relation2num[rel]))
def test(model,data,test_data,device,opt):
    ''' Start training '''
    global max_entity_len
    entity_nums=len(data['kb']['id2entity'])
    repl_entity=list(range(entity_nums))
    print(entity_nums)
    repl_entity_words=[entity2words[m] if m in entity2words else [0] for m in repl_entity]
    repl_entity_len=[len(m) for m in repl_entity_words]

    # repl_entity=torch.LongTensor(repl_entity).to(device)
    # repl_entity_words=torch.LongTensor(repl_entity_words).to(device)
    # repl_entity_len=torch.Tensor(repl_entity_len).to(device)

    mean_rank = [0, 0, 0, 0]
    hits10 = [0, 0, 0, 0]
    hits3=[0,0,0,0]
    hits1=[0,0,0,0]
    relation2mean_rank = {}
    relation2hits10 = {}
    relation2num = {}

    with torch.no_grad():
        start_time = time.time()
        count=0
        for batch in tqdm(test_data, mininterval=2, desc='  - (Testing)   ', leave=False):
            words, words_len, triples_id, triples_words, triples_len= batch
            words=words.to(device)
            words_len=words_len.to(device)
            # print(triples_id)
            # print(triples_id[0])
            #
            # print(triples_words)
            # print(triples_len)
            # print(triple_words.size())

            start=0
            f=False

            repl_head_result=[]
            repl_tail_result=[]
            hr=triples_words[0][0]+[Constants.ER]+triples_words[0][1]+[Constants.RE]
            rt=[Constants.ER]+triples_words[0][1]+[Constants.RE]+triples_words[0][2]
            while start<entity_nums:
                end=min(start+opt.test_batch_size,entity_nums)

                # 替换头实体
                triples_id_head=[]
                triples_len_head=[]
                triples_sen_head=[]
                max_head_len=0
                for repl_id,l,repl_words in zip(repl_entity[start:end],repl_entity_len[start:end],repl_entity_words[start:end]):
                    triples_id_head.append([repl_id,triples_id[0][1],triples_id[0][2]])
                    triples_len_head.append([l,triples_len[0][1],triples_len[0][2]])
                    triples_sen_head.append(repl_words+rt)
                    max_head_len=max(len(triples_sen_head[-1]),max_head_len)
                triples_sen_head_padded=[m+[Constants.PAD]*(max_head_len-len(m)) for m in triples_sen_head]
                #print(triples_id_head)
                triples_id_head=torch.LongTensor(triples_id_head).to(device)
                triples_sen_head_padded=torch.LongTensor(triples_sen_head_padded).to(device)
                triples_len_head=torch.LongTensor(triples_len_head).to(device)

                if not f:
                    score_head, enc_text=model(words,words_len,triples_id_head,triples_sen_head_padded,triples_len_head,None,'pos',True)
                    f=True
                else:
                    score_head, enc_text=model(words,words_len,triples_id_head,triples_sen_head_padded,triples_len_head,enc_text,'neg',True)
                #print(score_head)
                del triples_id_head
                del triples_sen_head
                del triples_len_head
                # 替换尾实体
                triples_id_tail=[]
                triples_len_tail=[]
                triples_sen_tail=[]
                max_tail_len=0
                for repl_id,l,repl_words in zip(repl_entity[start:end],repl_entity_len[start:end],repl_entity_words[start:end]):
                    triples_id_tail.append([triples_id[0][0],triples_id[0][1],repl_id])
                    triples_len_tail.append([triples_len[0][0],triples_len[0][1],l])
                    triples_sen_tail.append(hr+repl_words)
                    max_tail_len=max(len(triples_sen_tail[-1]),max_tail_len)
                triples_sen_tail_padded=[m+[Constants.PAD]*(max_tail_len-len(m)) for m in triples_sen_tail]
                #print(triples_id_tail)
                triples_id_tail = torch.LongTensor(triples_id_tail).to(device)
                triples_sen_tail_padded = torch.LongTensor(triples_sen_tail_padded).to(device)
                triples_len_tail = torch.LongTensor(triples_len_tail).to(device)

                score_tail, enc_text=model(words, words_len, triples_id_tail, triples_sen_tail_padded,triples_len_tail,enc_text,'neg',True)

                del triples_id_tail
                del triples_sen_tail
                del triples_len_tail

                repl_head_result.append(score_head)
                repl_tail_result.append(score_tail)

                start=end
                del score_head
                del score_tail

            score_head = torch.cat(repl_head_result, 0)
            score_tail = torch.cat(repl_tail_result, 0)

            _, index_head = torch.sort(score_head)
            _, index_tail = torch.sort(score_tail)

            del score_head
            del score_tail
            del words
            del words_len

            triple=triples_id[0]
            rank_head = np.where(index_head.cpu().numpy() == triple[0])[0][0]
            rank_tail = np.where(index_tail.cpu().numpy() == triple[2])[0][0]

            del index_head
            del index_tail
            #print(rank_head,rank_tail)

            if triple[1]not in relation2mean_rank:
                relation2mean_rank[triple[1]] = [0, 0, 0, 0]
                relation2hits10[triple[1]] = [0, 0, 0, 0]
                relation2num[triple[1]] = 1
            else:
                relation2num[triple[1]] += 1

            mean_rank[0] += (rank_head + 1)
            mean_rank[1] += (rank_tail + 1)
            relation2mean_rank[triple[1]][0] += (rank_head + 1)
            relation2mean_rank[triple[1]][1] += (rank_tail + 1)

            if rank_head + 1 <= 10:
                hits10[0] += 1
                relation2hits10[triple[1]][0] += 1
                if rank_head+1<=3:
                    hits3[0]+=1
                    if rank_head==0:
                        hits1[0]+=1
            if rank_tail + 1 <= 10:
                hits10[1] += 1
                relation2hits10[triple[1]][1] += 1
                if rank_tail+1<=3:
                    hits3[1]+=1
                    if rank_tail==0:
                        hits1[1]+=1

            count+=1
            if count%100==0:
                print('test: {nums:d}, cost: {cost:3.3f} min'.format(nums=count,cost=(time.time() - start_time) / 60))
                print('mean_rank:{:.1f}, hits@10:{:.4f}, hits@3:{:.4f}, hits@1:{:.4f}'.format((mean_rank[0]+mean_rank[1])/(2*count),
                    (hits10[0]+hits10[1])/(2*count),(hits3[0]+hits3[1])/(2*count),(hits1[0]+hits1[1])/(2*count)))
                start_time=time.time()
            if count%1000==0:
                print_all_relation(mean_rank,hits10,hits3,hits1,relation2mean_rank,relation2hits10,relation2num,data,count)

            torch.cuda.empty_cache()
        print('mean_rank:{:.1f}, hits@10:{:.4f}, hits@3:{:.4f}, hits@1:{:.4f}'.format(
            (mean_rank[0] + mean_rank[1]) / (2 * count),
            (hits10[0] + hits10[1]) / (2 * count), (hits3[0] + hits3[1]) / (2 * count),
            (hits1[0] + hits1[1]) / (2 * count)))
        print_all_relation(mean_rank,hits10,hits3,hits1,relation2mean_rank,relation2hits10,relation2num,data,count)

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-log', default=None)
    parser.add_argument('-no_cuda', action='store_true',default=False)
    parser.add_argument('-test_batch_size',default=500)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    '''========= Loading Dataset ========='''
    data = pickle.load(open('../dataset_en/all_data.pkl','rb'))
    emb_matrix=data['dict']['emb_matrix']

    global entity2words
    entity2words = data['kb']['entity_id2words_id']

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

    #path='checkpoint/2019-06-06_17:36:38'
    path='checkpoint/2019-07-08_01:09:30'
    model_path=path+'/model300'
    print('load model from',model_path)
    checkpoint=torch.load(model_path)
    transformer.load_state_dict(checkpoint['model'])

    #opt=checkpoint['settings']
    print('train opt',checkpoint['settings'])
    print('test opt',opt)
    transformer.eval()
    test(transformer,data,test_data, device,opt)

def prepare_dataloaders(data):
    # ========= Preparing DataLoader =========#
    test_loader = torch.utils.data.DataLoader(
        KBDataset(
            words=data['test']['sen_id'],
            triples=data['test']['triple_id'],
            triples_words=data['test']['triple_words_id'],
            triple_len=data['test']['triple_len'],
            ),
        num_workers=1,
        batch_size=1,
        collate_fn= paired_collate_fn,
        shuffle=False)
    return test_loader


def paired_collate_fn(insts):
    words,triples_id,triples_words,triples_len=list(zip(*insts))

    # 文本pad
    words_len = [len(sen) for sen in words]
    max_words_len = max(words_len)
    words_padded = np.array([sen + [Constants.PAD] * (max_words_len - len(sen)) for sen in words])
    # triples_sen=[tri_words[0] + [Constants.ER] + tri_words[1] + [Constants.RE] + tri_words[2] for tri_words in triples_words]
    # max_triples_sen_len=max([len(tri_sen) for tri_sen in triples_sen])
    #
    # triples_sen_padded=[tri_words + [Constants.PAD] * (max_triples_sen_len - len(tri_words)) for tri_words in triples_sen]

    batch_words=torch.LongTensor(words_padded)
    batch_words_len=torch.LongTensor(words_len)
    # batch_triples_id=torch.LongTensor(triples_id)
    # batch_triples_words=torch.LongTensor(triples_words)
    # batch_triples_len=torch.LongTensor(triples_len)

    return batch_words, batch_words_len, \
           triples_id, triples_words, triples_len

if __name__ == '__main__':
    main()

