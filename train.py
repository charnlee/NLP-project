# 导入所需模块
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
from model import BertClassifier
from dataset import Dataset
import os
# 训练
def train(model, train_dataset, val_dataset, epochs):
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optim = Adam(model.parameters(), lr=1e-5)
    best_avg_acc_val = 0

    for epoch in range(epochs):
        model.train()
        total_loss_train = total_acc_train = 0
        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
        for train_input, train_label in train_iterator:
            train_label = train_label.to(device)
            attention_mask = train_input['attention_mask'].squeeze(1).to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, train_label)
            total_loss_train += loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        total_loss_val = total_acc_val = 0
        val_iterator = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Validate]', leave=False)
        with torch.no_grad():
            for val_input, val_label in val_iterator:
                val_label = val_label.to(device)
                attention_mask = val_input['attention_mask'].squeeze(1).to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_ids, attention_mask)
                loss = criterion(output, val_label)
                total_loss_val += loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        avg_acc_val = total_acc_val / len(val_dataset)
        if avg_acc_val > best_avg_acc_val:
            best_avg_acc_val = avg_acc_val
            save_path = "model/finetune_model"
            directory = os.path.dirname(save_path)
            # 如果目录不存在，创建它
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(model.state_dict(), save_path)

        print(f'Epoch {epoch + 1}: Train Loss {total_loss_train / len(train_dataset):.3f}, '
              f'Train Acc {total_acc_train / len(train_dataset):.3f}, '
              f'Val Loss {total_loss_val / len(val_dataset):.3f}, '
              f'Val Acc {avg_acc_val:.3f}')

if __name__ == "__main__":

    train_dataset = pd.read_excel('data/ChnSentiCorp_htl_all.xlsx')
    dataset = Dataset(train_dataset, "bert-base-chinese")
    train_data, val_data = random_split(dataset, [round(0.8 * len(train_dataset)), round(0.2 * len(train_dataset))])
    model = BertClassifier("bert-base-chinese",0.5,freeze_bert=True)
    print('训练开始')
    train(model,train_dataset=train_data,val_dataset=val_data,epochs=10)
    print('训练结束')





