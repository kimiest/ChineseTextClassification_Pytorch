import json
from tqdm import tqdm
from random import shuffle
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# *************************************
# 读取.json格式的原始数据
# *************************************
def read_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    shuffle(data)
    k = 800
    return data[:k], data[k:]  # 前k个样本用于开发验证，剩下的用于训练


# ************************************
# 用Dataset类封装数据
# ************************************
class MyDataset(Dataset):
    def __init__(self, data, Config):
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)

        self.input_ids = tokenizer([x['text'] for x in data],
                                   truncation=True, max_length=Config.max_length,
                                   padding='max_length', return_tensors='pt')['input_ids']
        self.labels = [x['type'] for x in data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_inputs = self.input_ids[idx]
        batch_labels = self.labels[idx]
        return {'inputs': batch_inputs, 'labels': batch_labels}


# *************************************
# 返回训练、开发验证数据DataLoader
# *************************************
def get_train_dev_DataLoader(Config):
    dev, train = read_json(Config.train_path)
    train_dl = DataLoader(MyDataset(train, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('训练数据加载完成')
    dev_dl = DataLoader(MyDataset(dev, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('开发验证数据加载完成')
    return train_dl, dev_dl


if __name__ == '__main__':
    # 测试数据集加载部分是否好使
    from config import Config
    tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
    train_dl, dev_dl = get_train_dev_DataLoader(Config)
    for batch in train_dl:
        print(batch['inputs'].shape, batch['labels'])
        break





