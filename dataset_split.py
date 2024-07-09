import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(file_path, train_ratio=0.8):
    # 加载数据
    data = pd.read_excel(file_path)

    # 拆分数据集
    train_data, test_data = train_test_split(data, train_size=train_ratio, random_state=42)

    # 保存到新的Excel文件
    train_data.to_excel('data/train_dataset.xlsx', index=False)
    test_data.to_excel('data/test_dataset.xlsx', index=False)

    print("Data split into training and testing sets successfully.")
    print(f"Training set size: {len(train_data)}, Testing set size: {len(test_data)}")


# 使用实际的文件路径
split_data('data/ChnSentiCorp_htl_all.xlsx')
