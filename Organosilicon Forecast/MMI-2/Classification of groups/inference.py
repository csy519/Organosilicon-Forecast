# -*- coding: utf-8 -*-

import pickle
import argparse
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(50, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 4)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=-1)

        return x


class MyDataset(Dataset):
    def __init__(self, inputs):
        super(MyDataset, self).__init__()
        self.inputs = inputs

    def __getitem__(self, idx):
        x = self.inputs[idx]

        return x

    def __len__(self):
        return len(self.inputs)


def inference(args):
    origin_data = pd.read_excel(args.data_path, header=None)

    scaler = pickle.load(open(args.scaler_path, 'rb'))
    X = scaler.transform(origin_data.values).astype(np.float32)

    dataset = MyDataset(X)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = MLP()
    model.load_state_dict(torch.load(args.ckpt_path))
    if use_cuda:
        model = model.to(device)

    total_pred = []
    with tqdm(total=len(dataloader), desc=f'Inference') as pbar:
        for i, x in enumerate(dataloader):
            x = x.to(device)
            output = model(x)
            total_pred.append(torch.argmax(output, dim=-1).cpu().numpy())
            pbar.update(1)

    total_pred = np.concatenate(total_pred, axis=0).astype(np.int)
    predict_df = pd.DataFrame(total_pred)
    results_df = pd.concat([origin_data, predict_df],axis=1)
    results_df.to_csv(args.output_path, index=None, header=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='your_file_name.xlsx', type=str, help='')
    parser.add_argument('--output_path', default='result.csv', type=str, help='')
    parser.add_argument('--ckpt_path', default='ckpt/model.pt', type=str, help='')
    parser.add_argument('--scaler_path', default='ckpt/scaler.pkl', type=str, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    args = parser.parse_args()

    inference(args)



