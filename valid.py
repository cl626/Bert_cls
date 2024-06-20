import process_imdb
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import logging
import numpy as np
import pickle
import os

# 使用BERT使其向量化

MAXLEN = 512 - 2
BATCHSIZE = 8

from transformers import AutoTokenizer,AutoModelForSequenceClassification

# 调用词向量化模型
path = "bert-base-uncased"
with open(os.path.join('model',path+'.pkl'),'rb') as f: 
    modelset = pickle.load(f)
tokenizer = modelset['tokenizer']
model = modelset['pre_net']


'''构建数据集和数据迭代器，设定 batch_size 大小为'''
input_labels2 = torch.unsqueeze(torch.tensor(process_imdb.test_labels), dim=1)
valid_union = tokenizer(process_imdb.train_samples,max_length=MAXLEN,padding='max_length',truncation=True,return_tensors='pt')
valid_set = TensorDataset(valid_union['input_ids'], valid_union['attention_mask'],
                         torch.LongTensor(input_labels2))
valid_loader = DataLoader(dataset=valid_set,
                         batch_size=BATCHSIZE, )

for i, (train, mask, label) in enumerate(valid_loader):
    print(train.shape, mask.shape, label.shape)  #
    break

'''预测函数，用于预测结果'''

def predict(logits):
    res = torch.argmax(logits, dim=1)  # 按行取每行最大的列下标
    return res

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def valid_model(net):
    net.eval()
    net = net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, mask, label) in enumerate(valid_loader):
            print("batch_idx=%d",batch_idx)
            logging.info("test batch_id=" + str(batch_idx))

            data, mask, label = data.to(device), mask.to(device), label.to(device)
            output = net(data, token_type_ids=None, attention_mask=mask)  # 调用model模型时不传入label值。
            # output的形式为（元组类型，第0个元素是每个batch中好评和差评的概率）
            # print(output[0],label)
            print(predict(output[0]), label.flatten())
            total += label.size(0)  # 逐次按batch递增
            correct += (predict(output[0]) == label.flatten()).sum().item()
        print(f"正确分类的样本数 {correct}，总数 {total},准确率 {100.*correct/total:.3f}%")

        
valid_model(model)