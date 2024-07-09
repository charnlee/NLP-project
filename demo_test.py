import pandas as pd
from transformers import BertTokenizer
import torch

# 创建一个简单的数据字典
data = {
    'review': ["This product is excellent!", "Worst experience ever."],
    'label': [1, 0]
}

# 指定模型路径，这里假设是 'bert-base-uncased'
model_path = 'bert-base-uncased'

# 初始化Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, datas, model_path):
        self.labels = datas['label']
        tokenizer = BertTokenizer.from_pretrained(model_path)
        self.reviews = [
            tokenizer(str(review), padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            for review in datas['review']]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

# 创建数据集实例
dataset = CustomDataset(data, model_path)

# 获取数据集中的第一个元素
first_sample = dataset[0]
print(first_sample)
# print("Input IDs:", first_sample[0]['input_ids'])
# print("Attention Mask:", first_sample[0]['attention_mask'])
# print("Label:", first_sample[1])
