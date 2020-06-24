import torch
import inspect
# t=torch.nn.LSTM(3,4)
# print(t.__dict__["_parameters"])
# for m in t.__dict__["_parameters"]:
#     print(t.__dict__["_parameters"][m].data.dtype)
#
# # t.weight_ih_l0.data=t.weight_ih_l0.data.type(torch.float16)
# for m in t.__dict__["_parameters"]:
#     t.__dict__["_parameters"][m].data=t.__dict__["_parameters"][m].data.type(torch.float16)
#
# for m in t.__dict__["_parameters"]:
#     print(t.__dict__["_parameters"][m].data.dtype)
#
# a=torch.tensor(5)
# print(a)
#
# tt=torch.nn.Linear(3,4)
# for m in tt.__dict__["_parameters"]:
#     print(tt.__dict__["_parameters"][m].data.dtype)
#
# word_emb = torch.nn.Embedding(2, 3,_weight=torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float16))
# for m in word_emb.__dict__["_parameters"]:
#     print(word_emb.__dict__["_parameters"][m].data.dtype)
#
# c=' 龙抬头 二月二月 '
# d=c.replace('二月','龙抬头')
# print(d)
#
# if sim_list[i] in s:
#     a = replace(sim_list[i], sim_list[0])
#     print(.....)

a=torch.tensor([[1,2,3],[4,5,6]])
b=torch.tensor([[2,3,4],[4,5,6]])
print(b[:,0:1])
print(a*b[:,0:1])

y = torch.tensor([-1],dtype=torch.float32)
print(y.requires_grad)