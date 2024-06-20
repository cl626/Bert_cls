import process_imdb
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch import optim
import logging
import numpy as np
import pickle,os

# 使用BERT使其向量化

MAXLEN = 512 - 2
BATCHSIZE = 8

from transformers import BertForSequenceClassification, AdamW, BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup

# 调用词向量化模型
path = "bert-base-uncased"
with open(os.path.join('model',path+'pkl'),'rb') as f: 
    modelset = pickle.load(f)
tokenizer = modelset['tokenizer']
model = modelset['pre_net']

params_dir='model/bert_base_model_beta.pkl'
model.load_state_dict(torch.load(params_dir))

# 把文本转为序号
def convert_text_to_ids(tokenizer, sentence, limit_size=MAXLEN):
    t = tokenizer.tokenize(sentence)[:limit_size]       # 分词
    encoded_ids = tokenizer.encode(t)                   # 转为编码/词序号
    if len(encoded_ids) < limit_size + 2:               # 不够长补齐
        tmp = [0] * (limit_size + 2 - len(encoded_ids))
        encoded_ids.extend(tmp)
    return encoded_ids


# 空格映射为0，对应的掩码赋值为0，其余为1
def get_att_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [int(float(i > 0)) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks

'''构建数据集和数据迭代器，设定 batch_size 大小为'''
input_ids2 = [convert_text_to_ids(tokenizer, sen) for sen in process_imdb.test_samples]
input_labels2 = torch.unsqueeze(torch.tensor(process_imdb.test_labels), dim=1)
atten_tokens_eval = get_att_masks(input_ids2)
test_set = TensorDataset(torch.LongTensor(input_ids2), torch.LongTensor(atten_tokens_eval),
                         torch.LongTensor(input_labels2))
test_loader = DataLoader(dataset=test_set,
                         batch_size=BATCHSIZE, )

for i, (train, mask, label) in enumerate(test_loader):
    print(train.shape, mask.shape, label.shape)  #
    break

'''预测函数，用于预测结果'''

def predict(logits):
    res = torch.argmax(logits, dim=1)  # 按行取每行最大的列下标
    return res

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(net):
    net.eval()
    net = net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, mask, label) in enumerate(test_loader):
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

        
test_model(model)