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
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
from model import RegressionModel
from datasets import RegressionDataset


def train(args):
    data = pd.read_excel(args.data_path).values

    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()
    X = scaler1.fit_transform(data[:, :14]).astype(np.float32)
    y = data[:, [14, 15]].astype(np.int32)
    y = scaler2.fit_transform(y).astype(np.float32)

    with open('checkpoints/regression_scaler1.pkl', 'wb') as f:
        pickle.dump(scaler1, f)

    with open('checkpoints/regression_scaler2.pkl', 'wb') as f:
        pickle.dump(scaler2, f)

    train_index, dev_index = train_test_split(list(range(len(X))), train_size=args.train_ratio, shuffle=True)
    X_train, X_dev = X[train_index], X[dev_index]
    y_train, y_dev = y[train_index], y[dev_index]

    train_dataset = RegressionDataset(X_train, y_train)
    dev_dataset = RegressionDataset(X_dev, y_dev)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = RegressionModel()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)

    best_mse = 0.
    best_r2_score = float('-inf')
    best_r2_score_list = []
    for epoch in range(args.epochs):
        total_loss_train = 0

        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}') as pbar:
            for i, (x, y) in enumerate(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                o = model(x)

                batch_loss = criterion(o, y)
                total_loss_train += batch_loss.item()

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_train / (i + 1))})
                pbar.update(1)

        total_loss_val = 0
        total_ground_truth = []
        total_pred = []

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'Evaluating') as pbar:
                for i, (x, y) in enumerate(val_dataloader):
                    total_ground_truth.append(y.cpu().numpy())

                    x = x.to(device)
                    y = y.to(device)
                    o = model(x)
                    batch_loss = criterion(o, y)

                    total_loss_val += batch_loss.item()
                    total_pred.append(o.cpu().numpy())

                    pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_val / (i + 1))})
                    pbar.update(1)

            total_ground_truth = np.concatenate(total_ground_truth, axis=0)
            total_pred = np.concatenate(total_pred, axis=0)
            r2_score_1 = r2_score(total_ground_truth, total_pred, multioutput='raw_values')
            mse = mean_squared_error(total_ground_truth, total_pred, multioutput='raw_values')
            mae = mean_absolute_error(total_ground_truth, total_pred, multioutput='raw_values')
            if np.sum(r2_score_1) >= best_r2_score:
                best_r2_score = np.sum(r2_score_1)
                best_r2_score_list = r2_score_1
                best_mse = mse
                best_mae = mae
                torch.save(model.state_dict(), 'checkpoints/regression_model.pt')
                
                print(f'Epoch {epoch}, best r2_score of regression task 1 is {best_r2_score}.')
                print(f'Epoch {epoch}, best r2_score list of regression task 1 is {best_r2_score_list}.')
                print(f'Epoch {epoch}, best mse of regression task 1 is {best_mse}.')
                print(f'Epoch {epoch}, best mae of regression task 1 is {best_mae}.')
        model.train()

    print('###########' * 5)
    print(f'The r2 score of regression task 1 is {best_r2_score_list}.')
    print(f'The mse of regression task 1 is {best_mse}.')
    print(f'The mae of regression task 1 is {best_mae}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/PBT模型-1016.xlsx', type=str, help='')
    parser.add_argument('--train_ratio', default=0.7, type=float, help='the ratio of train set')
    parser.add_argument('--epochs', default=1000, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()

    train(args)
