# 说明：
* make_pretrain,bert_train,bert_test是用于基于bert-base-uncased预训练的
* 测试模型只要两步：

1. make_valid中修改path的值并执行```python make_valid.py```

![image-20240620085637450](./assets/image-20240620085637450.png)

* 会在当前model目录下生成一个模型（要用代理，所以在本地进行)

2. valid.py中修改path的名字为前面模型"/"后的模型名，并运行

![image-20240620085844583](./assets/image-20240620085844583.png)
