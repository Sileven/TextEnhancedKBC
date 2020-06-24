import argparse
import sys
sys.path.append('../')
import time
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import dataset_en.Constants as Constants
from dataset_en.dataset import KBDataset
from trans_mul_cat import Trans
import numpy as np
import pickle
import os
import random
from Optim import ScheduledOptim
# import inspect
# from gpu_mem_track import MemTracker  # 引用显存跟踪代码
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tensor_type=torch.float32

# frame = inspect.currentframe()
# gpu_tracker = MemTracker(frame)      # 创建显存检测对象


def train(model, training_data, optimizer, device, opt, path, start, scheduler):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')
    if path=='':
        time_now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path=os.path.join('checkpoint',time_now)
    else:
        save_path=path
    print('save model to',save_path)
    for epoch_i in range(start,opt.epoch):
        start = time.time()

        train_loss=0
        count=0
        scheduler.step()
        for batch in tqdm(
                training_data, mininterval=2,
                desc='  - (Training)   ', leave=False):
            words, words_len, \
            triples_pos_id, triples_pos_sen, triples_pos_sen_len, \
            triples_neg_id, triples_neg_sen, triples_neg_sen_len, \
            triples_neg_rel_id, triples_neg_rel_sen, triples_neg_rel_sen_len = map(lambda x: x.to(device), batch)

            # gpu_tracker.track()
            """ text[batch_size,max_text_len]
                text_len[batch_size]
                triple_id[batch_size,3]
                triple_words[batch_size,max_triple_len]
                triple_len[batch_size,3]
            """
            optimizer.zero_grad()
            loss_pos, enc_text= model(words,words_len,triples_pos_id,triples_pos_sen,triples_pos_sen_len,None, 'pos',False)
            # del triples_pos_id
            # del triples_pos_sen
            # del triples_pos_sen_len
            # gpu_tracker.track()
            loss_neg, enc_text= model(words,words_len,triples_neg_id,triples_neg_sen,triples_neg_sen_len,enc_text, 'neg',False)
            loss_neg_rel, enc_text = model(words,words_len,triples_neg_rel_id,triples_neg_rel_sen, triples_neg_rel_sen_len, enc_text, 'neg', False)

            # del words
            # del words_len
            # del triples_neg_id
            # del triples_neg_sen
            # del triples_neg_sen_len
            # del enc_text
            # gpu_tracker.track()
            if opt.cuda:
                y = torch.tensor([-1],dtype=tensor_type).cuda()
            else:
                y = torch.tensor([-1],dtype=tensor_type)

            # print(loss_pos.size())
            # print(loss_neg.size())
            loss_en = nn.functional.margin_ranking_loss(loss_pos, loss_neg, y, margin=opt.margin, reduction='sum')
            loss_rel = nn.functional.margin_ranking_loss(loss_pos, loss_neg_rel, y, margin=opt.margin, reduction='sum')
            loss = loss_en + loss_rel
            #print(loss)
            # gpu_tracker.track()
            loss.backward()
            # update parameters
            optimizer.step()
            #optimizer.step_and_update_lr()
            # gpu_tracker.track()

            # note keeping
            train_loss += loss.item()
            # print(loss.item())
            # del loss
            # del loss_pos
            # del loss_neg
            # gpu_tracker.track()
            torch.cuda.empty_cache()
        print('[ Epoch', epoch_i, ']- (Training)  loss: {loss:3.3f}, cost: {cost:3.3f} min, learning_rate: {lr:.8f}'
              .format(loss=train_loss,cost=(time.time() - start) / 60, lr=optimizer.state_dict()['param_groups'][0]['lr']))

        # 保存模型
        if (epoch_i+1)%opt.save_every==0:
            model_state_dict = model.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'settings': opt,
                'epoch': epoch_i}
                #'step':optimizer.n_current_steps}

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model_name = 'model' + str(epoch_i + 1)
            torch.save(checkpoint, os.path.join(save_path, model_name))
            print("\nsave as", model_name, '\n')
            first_model = (epoch_i + 1) - opt.max_save * opt.save_every
            if first_model > 0:
                os.remove(os.path.join(save_path, 'model' + str(first_model)))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-d_word_vec', type=int, default=256)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-dropout', type=float, default=0)
    parser.add_argument('-n_warmup_steps',type=float, default=400)
    parser.add_argument('-log', default=None)

    parser.add_argument('-no_cuda', action='store_true',default=False)
    parser.add_argument('-margin',default=8)
    parser.add_argument('-learning_rate',default=0.001)
    parser.add_argument('-save_every',default=10)
    parser.add_argument('-max_save',default=3)
    parser.add_argument('-train_new',default=True)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    '''========= Loading Dataset ========='''
    data = pickle.load(open('../dataset_en/all_data.pkl','rb'))
    emb_matrix=data['dict']['emb_matrix']

    global entity2words
    global triple_pool
    global entity_pool
    global id2relation
    global relation2words
    entity2words = data['kb']['entity_id2words_id']
    triple_pool = data['kb']['triple_pool']
    entity_pool = data['kb']['entity_pool']
    id2relation = data['kb']['id2relation']
    relation2words = data['kb']['relation_id2words_id']

    training_data = prepare_dataloaders(data, opt)
    opt.vocab_size = emb_matrix.shape[0]

    device = torch.device('cuda' if opt.cuda else 'cpu')
    transformer = Trans(
        emb_matrix,
        len(data['kb']['id2entity']),
        len(data['kb']['id2relation']),
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        dropout=opt.dropout).to(device)

    # gpu_tracker.track()
    start_epoch=0
    path=''
    if not opt.train_new:
        path='checkpoint/2019-05-23_16:22:54'
        model_path=path+'/model20'
        print('load model from',model_path)
        checkpoint=torch.load(model_path)
        transformer.load_state_dict(checkpoint['model'])
        opt=checkpoint['settings']
        # opt.learning_rate=0.005*pow(0.8,19)
        opt.learning_rate = 0.0032
        start_epoch=checkpoint['epoch']+1

    print(opt)
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, transformer.parameters()),lr=opt.learning_rate,betas=(0.9, 0.98), eps=1e-09)
    optimizer=optim.SGD(filter(lambda x: x.requires_grad, transformer.parameters()),lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    #optimizer = ScheduledOptim(
        #optim.SGD(filter(lambda x: x.requires_grad, transformer.parameters()),opt.learning_rate),opt.d_model, opt.n_warmup_steps,cur_steps)

    para = sum([np.prod(list(p.size())) for p in transformer.parameters()])
    print('Model {} : params: {:4f}M'.format(transformer._get_name(), para * 2 / 1000 / 1000))

    train(transformer, training_data, optimizer, device ,opt,path,start_epoch,scheduler)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        KBDataset(
            words=data['train']['sen_id'],
            triples=data['train']['triple_id'],
            triples_words=data['train']['triple_words_id'],
            triple_len=data['train']['triple_len'],
            ),
        num_workers=1,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    # valid_loader = torch.utils.data.DataLoader(
    #     KBDataset(
    #         words=data['test']['sen_id'],
    #         triples=data['test']['triple_id'],
    #         triples_words=data['test']['triple_words_id'],
    #         triple_len=data['test']['triple_len'],
    #         ),
    #     num_workers=2,
    #     batch_size=opt.batch_size,
    #     collate_fn=paired_collate_fn)
    return train_loader


# 构造负样例
def paired_collate_fn(insts):
    words,triples_pos_id,triples_pos_words,triples_pos_len=list(zip(*insts))
    # 文本pad
    words_len=[len(sen) for sen in words]
    max_words_len=max(words_len)
    words_padded=np.array([sen + [Constants.PAD] * (max_words_len - len(sen)) for sen in words])

    # 构造负样例

    # 实体负样例
    triples_neg_id = []
    triples_neg_sen = []
    triples_neg_len = []
    max_triples_neg_sen_len = 0

    # 关系负样例
    triples_neg_rel_id=[]
    triples_neg_rel_sen = []
    triples_neg_rel_len = []
    max_triples_neg_rel_sen_len = 0

    # 正样例
    triples_pos_sen=[]
    max_triples_pos_sen_len=0

    for tri, tri_words, l in zip(triples_pos_id, triples_pos_words, triples_pos_len):
        # 正样例
        triples_pos_sen.append(tri_words[0] + [Constants.ER] + tri_words[1] + [Constants.RE] + tri_words[2])
        sum_now=sum(list(l))+2
        if sum_now>max_triples_pos_sen_len:
            max_triples_pos_sen_len=sum_now

        # 负样例
        ## 对实体负采样
        ran_now = random.choice([0, 1])
        # 替换头
        if ran_now == 0:
            while 1:
                neg = random.sample(entity_pool, 1)[0]
                if neg!=tri[0] and (neg, tri[1], tri[2]) not in triple_pool:
                    break
            neg_words=entity2words[neg]
            triples_neg_id.append([neg, tri[1], tri[2]])
            triples_neg_sen.append(neg_words + [Constants.ER] + tri_words[1] + [Constants.RE] + tri_words[2])
            triples_neg_len.append([len(neg_words),len(tri_words[1]),len(tri_words[2])])
        # 替换尾
        else:
            while 1:
                neg = random.sample(entity_pool, 1)[0]
                if neg != tri[2] and (tri[0],tri[1],neg) not in triple_pool:
                    break
            neg_words=entity2words[neg]
            triples_neg_id.append([tri[0], tri[1], neg])
            triples_neg_sen.append(tri_words[0] + [Constants.ER] + tri_words[1] + [Constants.RE] + neg_words)
            triples_neg_len.append([len(tri_words[0]),len(tri_words[1]),len(neg_words)])

        sum_now=sum(triples_neg_len[-1])+2
        if sum_now > max_triples_neg_sen_len:
            max_triples_neg_sen_len = sum_now

        ## 对关系负采样
        neg = tri[1]
        while 1:
            neg = random.randint(0,len(id2relation)-1)
            if neg != tri[1] and neg in relation2words and (tri[0],neg,tri[2]) not in triple_pool:
                break
        neg_words=relation2words[neg]
        triples_neg_rel_id.append([tri[0],neg,tri[2]])
        triples_neg_rel_sen.append(tri_words[0]+ [Constants.ER] + neg_words + [Constants.RE] + tri_words[2])
        triples_neg_rel_len.append([len(tri_words[0]),len(neg_words),len(tri_words[2])])

        sum_now=sum(triples_neg_rel_len[-1])+2
        if sum_now > max_triples_neg_rel_sen_len:
            max_triples_neg_rel_sen_len = sum_now

    # 三元组正样例pad
    triples_pos_sen_padded=[tri_words + [Constants.PAD] * (max_triples_pos_sen_len - len(tri_words))
                            for tri_words in triples_pos_sen]
    triples_neg_sen_padded=[tri_words + [Constants.PAD] * (max_triples_neg_sen_len - len(tri_words))
                            for tri_words in triples_neg_sen]
    triples_neg_rel_sen_padded = [tri_words + [Constants.PAD] * (max_triples_neg_rel_sen_len - len(tri_words))
                                  for tri_words in triples_neg_rel_sen]

    batch_words=torch.LongTensor(words_padded)
    batch_words_len=torch.LongTensor(words_len)
    #print(triples_pos_id)
    batch_triples_pos_id=torch.LongTensor(triples_pos_id)
    batch_triples_pos_sen=torch.LongTensor(triples_pos_sen_padded)
    batch_triples_pos_sen_len=torch.LongTensor(triples_pos_len)

    batch_triples_neg_id=torch.LongTensor(triples_neg_id)
    batch_triples_neg_sen=torch.LongTensor(triples_neg_sen_padded)
    batch_triples_neg_sen_len=torch.LongTensor(triples_neg_len)

    batch_triples_neg_rel_id = torch.LongTensor(triples_neg_rel_id)
    batch_triples_neg_rel_sen = torch.LongTensor(triples_neg_rel_sen_padded)
    batch_triples_neg_rel_sen_len = torch.LongTensor(triples_neg_rel_len)

    return batch_words, batch_words_len, \
           batch_triples_pos_id, batch_triples_pos_sen, batch_triples_pos_sen_len, \
           batch_triples_neg_id, batch_triples_neg_sen, batch_triples_neg_sen_len, \
           batch_triples_neg_rel_id, batch_triples_neg_rel_sen, batch_triples_neg_rel_sen_len


if __name__ == '__main__':
    main()
