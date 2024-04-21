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
    def __init__(self, inputs, labels):
        super(MyDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]

        return x, y

    def __len__(self):
        return len(self.inputs)


def train(args):
    data = pd.read_excel(args.data_path, header=None).values

    scaler = StandardScaler()
    X = scaler.fit_transform(data[:, :50]).astype(np.float32)
    y = data[:, 50].astype(np.long)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=args.train_ratio)
    with open('ckpt/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    if use_cuda:
        model = model.to(device)
        criterion = criterion.to(device)

    best_acc = 0
    for epoch in range(args.epochs):
        total_loss_train = 0

        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}') as pbar:
            for i, (x, y) in enumerate(train_dataloader):
                x = x.to(device)
                y = y.to(device).long()
                output = model(x)
                batch_loss = criterion(output, y)
                total_loss_train += batch_loss.item()

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_train / (i + 1))})
                pbar.update(1)

        total_loss_val = 0
        total_ground_truth = []
        total_pred = []
        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'Evaluating') as pbar:
                for i, (x, y) in enumerate(val_dataloader):
                    total_ground_truth.append(y.cpu().numpy())

                    x = x.to(device)
                    y = y.to(device).long()

                    output = model(x)
                    batch_loss = criterion(output, y)
                    total_loss_val += batch_loss.item()
                    total_pred.append(torch.argmax(output, dim=-1).cpu().numpy())

                    pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_val / (i + 1))})
                    pbar.update(1)

            total_ground_truth = np.concatenate(total_ground_truth, axis=0)
            total_pred = np.concatenate(total_pred, axis=0)
            acc = accuracy_score(total_ground_truth, total_pred)

            if acc >= best_acc:
                best_acc = acc
                cm = confusion_matrix(total_ground_truth, total_pred, labels=[0, 1, 2, 3])
                print(f'Epoch {epoch}, best accuracy is {best_acc}.')
                torch.save(model.state_dict(), 'ckpt/model.pt')
    print('###########' * 5)
    print(f'The best accuracy is {best_acc}.')

    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax)

    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/data_muti1.xlsx', type=str, help='')
    parser.add_argument('--train_ratio', default=0.7, type=float, help='the ratio of train set')
    parser.add_argument('--epochs', default=1000, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='')
    args = parser.parse_args()

    train(args)



