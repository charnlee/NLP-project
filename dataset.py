from transformers import BertTokenizer
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, model_path):
        self.labels = datas['label']
        tokenizer = BertTokenizer.from_pretrained(model_path)
        self.reviews = [
            tokenizer(str(review), padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            for review in datas['review']
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.reviews[idx], np.array(self.labels[idx])
