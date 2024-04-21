# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, roc_auc_score, precision_score, f1_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle


class MLP(nn.Module):
    def __init__(self, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.layer1 = nn.Linear(45, 300)
        self.layer2 = nn.Linear(300, 500)
        self.layer3 = nn.Linear(500, 500)
        self.layer4 = nn.Linear(500, 100)
        self.layer5 = nn.Linear(100, 5)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.layer2(x))
        x = self.dropout2(x)
        x = F.leaky_relu(self.layer3(x))
        x = self.dropout3(x)
        x = F.leaky_relu(self.layer4(x))
        x = self.dropout4(x)
        x = self.layer5(x)

        return x


class MyDataset(Dataset):
    def __init__(self, inputs, labels_1):
        super(MyDataset, self).__init__()
        self.inputs = inputs
        self.labels_1 = labels_1

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels_1[idx]

        return x, y

    def __len__(self):
        return len(self.inputs)


def train(args):
    data = pd.read_excel(args.data_path, header=None).values

    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()
    X = scaler1.fit_transform(data[:, :45].astype(np.float32))
    y = data[:, [45, 46, 47, 48, 49]].astype(np.float32)
    y = scaler2.fit_transform(y)
    y1 = scaler2.inverse_transform(y)


    with open('checkpoints/regression_scaler1.pkl', 'wb') as f:
        pickle.dump(scaler1, f)

    with open('checkpoints/regression_scaler2.pkl', 'wb') as f:
        pickle.dump(scaler2, f)

    train_index, dev_index = train_test_split(list(range(len(X))), train_size=args.train_ratio, shuffle=True)
    X_train, X_dev = X[train_index], X[dev_index]
    y1_train, y1_dev = y[train_index], y[dev_index]

    train_dataset = MyDataset(X_train, y1_train)
    dev_dataset = MyDataset(X_dev, y1_dev)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = MLP()
    criterion1 = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=args.lr)

    if use_cuda:
        model = model.to(device)
        criterion1 = criterion1.to(device)

    best_r2_score_1 = float('-inf')
    best_r2_score_list = []
    for epoch in range(args.epochs):
        total_loss_train = 0

        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}') as pbar:
            for i, (x, y) in enumerate(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                o = model(x)

                batch_loss = criterion1(o, y)
                total_loss_train += batch_loss.item()

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_train / (i + 1))})
                pbar.update(1)

        total_loss_val = 0
        total_ground_truth1 = []
        total_pred1 = []

        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'Evaluating') as pbar:
                for i, (x, y) in enumerate(val_dataloader):
                    total_ground_truth1.append(y.cpu().numpy())

                    x = x.to(device)
                    y = y.to(device)

                    o = model(x)
                    batch_loss = criterion1(o, y)

                    total_loss_val += batch_loss.item()
                    total_pred1.append(o.cpu().numpy())

                    pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_val / (i + 1))})
                    pbar.update(1)

            total_ground_truth1 = np.concatenate(total_ground_truth1, axis=0)
            total_pred1 = np.concatenate(total_pred1, axis=0)
            mse = mean_squared_error(total_ground_truth1, total_pred1, multioutput='raw_values')
            mae = mean_absolute_error(total_ground_truth1, total_pred1, multioutput='raw_values')



            score = r2_score(total_ground_truth1, total_pred1, multioutput='raw_values')
            a = 1

            if np.sum(score) >= best_r2_score_1:
                best_r2_score_1 = np.sum(score)
                best_r2_score_list = score
                best_mse_score_list = mse
                best_mae_score_list = mae
                torch.save(model.state_dict(), 'checkpoints/regression_model.pt')
                total_pred = total_pred1
                total_truth = total_ground_truth1
                np.savetxt('data_pred.csv', total_pred, delimiter=',')
                np.savetxt('data_true.csv', total_truth, delimiter=',')
                print(f'Epoch {epoch}, best r2_score of regression task 1 is {best_r2_score_1}.')
                print(f'Epoch {epoch}, best r2_score list of regression task 1 is {best_r2_score_list}.')
                print(f'Epoch {epoch}, best mse_score list of regression task 1 is {best_mse_score_list}.')
                print(f'Epoch {epoch}, best mae_score list of regression task 1 is {best_mae_score_list}.')

    print('###########' * 5)
    print(f'The r2 score of regression task 1 is {best_r2_score_list}.')
    print(f'The mse score of regression task 1 is {best_mse_score_list}.')
    print(f'The mae score of regression task 1 is {best_mae_score_list}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data.xlsx', type=str, help='')
    parser.add_argument('--train_ratio', default=0.7, type=float, help='the ratio of train set')
    parser.add_argument('--epochs', default=2000, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    args = parser.parse_args()

    train(args)
