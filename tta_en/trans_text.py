# -*- coding: utf-8 -*-
"""只利用文本的模型"""
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../')
import dataset.Constants as Constants

tensor_type=torch.float16
def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def type_convert(t):
    for m in t.__dict__["_parameters"]:
        t.__dict__["_parameters"][m].data = t.__dict__["_parameters"][m].data.type(tensor_type)

class Trans(nn.Module):
    def __init__(
            self,
            emb_matrix, n_entity, n_relation,
            d_word_vec=256, d_model=256, dropout=0.1):
        super().__init__()

        self.word_emb = nn.Embedding(emb_matrix.shape[0],emb_matrix.shape[1],_weight=torch.tensor(emb_matrix,dtype=tensor_type))

        self.lstm_text=nn.LSTM(d_word_vec,d_model//2,1,batch_first=True,dropout=dropout,bidirectional=True)
        self.lstm_tri=nn.LSTM(d_word_vec,d_model//2,1,batch_first=True,bidirectional=True)
        self.lstm_tri_att=nn.LSTM(2*d_model,d_model//2,1,batch_first=True,bidirectional=True)

        type_convert(self.lstm_text)
        type_convert(self.lstm_tri)
        type_convert(self.lstm_tri_att)

        W1=torch.Tensor(d_model,d_model)
        torch.nn.init.xavier_uniform_(W1)
        self.W1 = nn.Parameter(W1.type(tensor_type))

        W2=torch.Tensor(d_model,d_model)
        torch.nn.init.xavier_uniform_(W2)
        self.W2 = nn.Parameter(W2.type(tensor_type))

        self.v=nn.Parameter(torch.Tensor(d_model).uniform_(-6/(d_model**0.5),6/(d_model**0.5)).type(tensor_type))

        self.entity_linear=nn.Linear(2*d_model,d_model)
        type_convert(self.entity_linear)

        self.relation_linear=nn.Linear(2*d_model,d_model)
        type_convert(self.relation_linear)

    def encode(self, text, text_len, lstm, emb_flag):
        if emb_flag:
            # [batch_size,max_text_len,d_word_vec]
            text_emb = self.word_emb(text)
        else:
            text_emb=text
        # print(text_emb.size())
        # 对实体描述按长度从大到小排序
        text_len_sorted, sorted_indices = torch.sort(text_len, descending=True)
        # 计算过后用于还原batch中数据顺序的indices
        _, desorted_indices = torch.sort(sorted_indices)
        # 排序过后的text
        text_emb_sorted = text_emb[sorted_indices]

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(text_emb_sorted, text_len_sorted, batch_first=True)
        lstm.flatten_parameters()
        # [batch_size,max_text_len,2*d_model]
        packed_output, (_, _) = lstm(packed_input)
        # print(packed_output)
        # print(packed_output.data.size())
        # print(packed_output.batch_sizes.size())
        padded_output=torch.nn.utils.rnn.pad_packed_sequence(packed_output,batch_first=True)
        # print(padded_output[0].size())
        # 还原原顺序
        output_desorted = torch.index_select(padded_output[0], 0, desorted_indices)
        # print('hey')
        return output_desorted

    def tri2text_att(self,enc_text,enc_triple,att_mask):
        # [batch_size,max_triple_len,max_text_len]
        att=(enc_text.matmul(self.W1).unsqueeze(1)+enc_triple.matmul(self.W2).unsqueeze(2)).matmul(self.v)

        return nn.functional.softmax(att.masked_fill(att_mask, -np.inf), dim=2)

    def batched_index_select(self, input, dim, index):
        views = [input.shape[0]] + \
                [1 if i != dim else -1 for i in range(1, len(input.shape))]
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, text, text_len, triple_id, triple_words, triple_len, enc_text_last, flag, test_flag):
        """ train:
            text[batch_size,max_text_len]
            text_len[batch_size]
            triple_id[batch_size,3]
            triple_words[batch_size,max_triple_len]
            triple_len[batch_size,3]

            test:
            text[1,len]
            text_len[1]
            triple_id[batch_size,3]
            triple_words[batch_size,max_triple_len]
            triple_len[batch_size,3]
        """

        triple_len_sum=torch.sum(triple_len,1)+2
        # time1=time.time()

        if flag=='pos':
            # 对句子编码[batch_size,max_text_len,2*d_model]
            # print(text.size())
            # print(text_len.size())
            enc_te = self.encode(text,text_len,self.lstm_text,True)
        else:
            enc_te = enc_text_last

        if test_flag:
            enc_text = enc_te.expand(triple_id.size(0), -1, -1)
        else:
            enc_text = enc_te
        # time2=time.time()

        # print(triple_words.size())
        # print(triple_len_sum.size())
        # 对三元组编码 [batch_size,max_triple_len,2*d_model]
        enc_triple=self.encode(triple_words,triple_len_sum,self.lstm_tri,True)
        # time3=time.time()
        # [batch_size,max_triple_len,max_text_len]
        att_mask = get_attn_key_pad_mask(text, triple_words)

        # 计算三元组对句子的编码
        # [batch_size,max_triple_len,max_text_len]
        att=self.tri2text_att(enc_text,enc_triple,att_mask)
        # [batch_size,max_triple_len,2*d_model]
        triple_text_rep=att.matmul(enc_text)

        # [batch_size,max_triple_len,2*d_model]
        enc_triple_final=self.encode(torch.cat((enc_triple,triple_text_rep),dim=2),triple_len_sum,self.lstm_tri_att, False)

        # 找到h和t的开始词和结束词
        # [batch_size]
        index_ht=triple_len[:,0]-1
        index_r1=index_ht+2
        index_rt=index_r1+triple_len[:,1]-1
        index_t1=index_rt+2
        index_tt=index_t1+triple_len[:,2]-1
        # [batch_size,2*d_model]
        h1=enc_triple_final[:,0,:]
        ht=self.batched_index_select(enc_triple_final,1,index_ht).squeeze()
        r1=self.batched_index_select(enc_triple_final,1,index_r1).squeeze()
        rt=self.batched_index_select(enc_triple_final,1,index_rt).squeeze()
        t1=self.batched_index_select(enc_triple_final,1,index_t1).squeeze()
        tt=self.batched_index_select(enc_triple_final,1,index_tt).squeeze()

        # [batch_size,d_model]
        h_text=self.entity_linear(torch.cat((h1,ht),dim=1))
        r_text=self.relation_linear(torch.cat((r1,rt),dim=1))
        t_text=self.entity_linear(torch.cat((t1,tt),dim=1))

        score=torch.sum(torch.abs(h_text + r_text - t_text), 1)

        return score,enc_te
