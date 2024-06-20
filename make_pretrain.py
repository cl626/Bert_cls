from transformers import BertTokenizer,BertForSequenceClassification
import pickle,os

# 更换为model中其他训练好的模型修改path
path = "google-bert/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(path)
pre_net = BertForSequenceClassification.from_pretrained(path)

bertset = {"tokenizer":tokenizer, "pre_net":pre_net}

# 从path取‘/’后面的字符，加上'.pkl'作为与预训练模型的名字
begin_pos = path.find(['/'])+1
model_name =  path[begin_pos:]+'.pkl'
with open(os.path.join('model',model_name),'wb') as f:
    pickle.dump(bertset,f)

