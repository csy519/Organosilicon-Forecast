# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class MLP(nn.Module):
    def __init__(self, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.dropout7 = nn.Dropout(dropout)

        self.layer1 = nn.Linear(43, 200)
        # self.layer2 = nn.Linear(300, 200)
        self.layer3 = nn.Linear(200, 200)

        # branch 1: classification task 1
        self.branch1_1 = nn.Linear(200, 100)
        self.branch1_2 = nn.Linear(100, 2)

        # branch 2: classification task 2
        self.branch2_1 = nn.Linear(200, 100)
        self.branch2_2 = nn.Linear(100, 2)

        # branch 3: regression task 1
        self.branch3_1 = nn.Linear(200, 500)
        self.branch3_2 = nn.Linear(500, 300)
        self.branch3_3 = nn.Linear(300, 200)
        self.branch3_4 = nn.Linear(200, 5)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = self.dropout1(x)
        # x = F.leaky_relu(self.layer2(x))
        # x = self.dropout(x)
        x = F.leaky_relu(self.layer3(x))
        x = self.dropout2(x)

        output1 = F.leaky_relu(self.branch1_1(x))
        output1 = self.dropout3(output1)
        output1 = F.softmax(self.branch1_2(output1), dim=-1)

        output2 = F.leaky_relu(self.branch2_1(x))
        output2 = self.dropout4(output2)
        output2 = F.softmax(self.branch2_2(output2), dim=-1)

        output3 = F.leaky_relu(self.branch3_1(x))
        output3 = self.dropout5(output3)
        output3 = F.leaky_relu(self.branch3_2(output3))
        output3 = self.dropout6(output3)
        output3 = self.branch3_3(output3)
        output3 = self.dropout7(output3)
        output3 = self.branch3_4(output3)

        return output1, output2, output3


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

    data = pd.read_excel(args.data_path, header=None)
    scaler1 = pickle.load(open(args.scaler_path, 'rb'))
    X = scaler1.transform(data.values).astype(np.float32)


    inference_dataset = MyDataset(X)
    inference_dataloader = DataLoader(inference_dataset, batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = MLP()
    model.load_state_dict(torch.load(args.ckpt_path))
    if use_cuda:
        model = model.to(device)

    total_pred1 = []
    total_pred2 = []
    with torch.no_grad():
        with tqdm(total=len(inference_dataloader), desc=f'Inference') as pbar:
            for x in inference_dataloader:
                x = x.to(device)
                o1, o2, _ = model(x)

                total_pred1.append(torch.argmax(o1, dim=-1).cpu().numpy())
                total_pred2.append(torch.argmax(o2, dim=-1).cpu().numpy())
                pbar.update(1)

    total_pred1 = np.concatenate(total_pred1, axis=0).astype(np.int32)
    total_pred2 = np.concatenate(total_pred2, axis=0).astype(np.int32)

    predict_df1 = pd.DataFrame(total_pred1)
    predict_df2 = pd.DataFrame(total_pred2)
    results_df = pd.concat([data, predict_df1, predict_df2], axis=1)

    results_df.to_csv(args.output_path, index=None, header=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/test.xlsx', type=str, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--ckpt_path', default='checkpoints/classification_model.pt', type=str, help='checkpoint path')
    parser.add_argument('--scaler_path', default='checkpoints/classification_scaler1.pkl', type=str, help='scaler path')
    parser.add_argument('--output_path', default='result/stage1_result.csv', type=str, help='output path')
    args = parser.parse_args()

    inference(args)



