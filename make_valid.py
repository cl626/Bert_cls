from transformers import AutoTokenizer,AutoModelForSequenceClassification
import pickle,os

# 更换为model中其他训练好的模型修改path
# path = "google-bert/bert-base-uncased"
path = "aypan17/roberta-base-imdb"
tokenizer = AutoTokenizer.from_pretrained(path)
net = AutoModelForSequenceClassification.from_pretrained(path)

model = {"tokenizer":tokenizer, "net":net}

# 从path取‘/’后面的字符，加上'.pkl'作为与预训练模型的名字
begin_pos = path.find("/")+1
model_name =  path[begin_pos:]+'.pkl'
with open(os.path.join('model',model_name),'wb') as f:
    pickle.dump(model,f)
