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
    def __init__(self, inputs, labels_1, labels_2, labels_3):
        super(MyDataset, self).__init__()
        self.inputs = inputs
        self.labels_1 = labels_1
        self.labels_2 = labels_2
        self.labels_3 = labels_3

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y1 = self.labels_1[idx]
        y2 = self.labels_2[idx]
        y3 = self.labels_3[idx]

        return x, y1, y2, y3

    def __len__(self):
        return len(self.inputs)


def train(args):
    data = pd.read_excel(args.data_path, header=None).values

    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()
    X = scaler1.fit_transform(data[:, :43]).astype(np.float32)
    y1 = data[:, 43].astype(np.int32)
    y2 = data[:, 44].astype(np.int32)
    y3 = data[:, [45, 46, 47, 48, 49]].astype(np.float32)
    y3 = scaler2.fit_transform(y3).astype(np.float32)

    with open('checkpoints/classification_scaler1.pkl', 'wb') as f:
        pickle.dump(scaler1, f)

    with open('checkpoints/classification_scaler2.pkl', 'wb') as f:
        pickle.dump(scaler2, f)

    train_index, dev_index = train_test_split(list(range(len(X))), train_size=args.train_ratio, shuffle=True)
    X_train, X_dev = X[train_index], X[dev_index]
    y1_train, y1_dev = y1[train_index], y1[dev_index]
    y2_train, y2_dev = y2[train_index], y2[dev_index]
    y3_train, y3_dev = y3[train_index], y3[dev_index]

    train_dataset = MyDataset(X_train, y1_train, y2_train, y3_train)
    dev_dataset = MyDataset(X_dev, y1_dev, y2_dev, y3_dev)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = MLP()
    criterion1 = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.1, 1]))
    criterion2 = nn.CrossEntropyLoss(weight=torch.FloatTensor([2., 1.]))
    criterion3 = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr=args.lr)

    if use_cuda:
        model = model.to(device)
        criterion1 = criterion1.to(device)
        criterion2 = criterion2.to(device)
        criterion3 = criterion3.to(device)

    best_acc1 = 0.
    best_acc2 = 0.
    best_auc1 = 0.
    best_auc2 = 0.
    scorebest_r2_score = float('-inf')
    best_r2_score_list = []
    for epoch in range(args.epochs):
        total_loss_train = 0

        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}') as pbar:
            for i, (x, y1, y2, y3) in enumerate(train_dataloader):
                x = x.to(device)
                y1 = y1.to(device)
                y2 = y2.to(device)
                y3 = y3.to(device)
                o1, o2, o3 = model(x)

                batch_loss1 = criterion1(o1, y1.long())
                batch_loss2 = criterion2(o2, y2.long())
                batch_loss3 = criterion3(o3, y3)

                batch_loss = batch_loss1 + 2 * batch_loss2 + 3 * batch_loss3
                total_loss_train += batch_loss.item()

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_train / (i + 1))})
                pbar.update(1)

        total_loss_val = 0
        total_ground_truth1 = []
        total_ground_truth2 = []
        total_ground_truth3 = []
        total_pred1 = []
        total_pred2 = []
        total_pred3 = []

        with torch.no_grad():
            with tqdm(total=len(val_dataloader), desc=f'Evaluating') as pbar:
                for i, (x, y1, y2, y3) in enumerate(val_dataloader):
                    total_ground_truth1.append(y1.cpu().numpy())
                    total_ground_truth2.append(y2.cpu().numpy())
                    total_ground_truth3.append(y3.cpu().numpy())

                    x = x.to(device)
                    y1 = y1.to(device)
                    y2 = y2.to(device)
                    y3 = y3.to(device)

                    o1, o2, o3 = model(x)
                    batch_loss1 = criterion1(o1, y1.long())
                    batch_loss2 = criterion2(o2, y2.long())
                    batch_loss3 = criterion3(o3, y3)

                    total_loss_val += (batch_loss1.item() + batch_loss2.item() + batch_loss3.item()) / 3
                    total_pred1.append(torch.argmax(o1, dim=-1).cpu().numpy())
                    total_pred2.append(torch.argmax(o2, dim=-1).cpu().numpy())
                    total_pred3.append(o3.cpu().numpy())

                    pbar.set_postfix({'loss': '{0:1.5f}'.format(total_loss_val / (i + 1))})
                    pbar.update(1)

            total_ground_truth1 = np.concatenate(total_ground_truth1, axis=0)
            total_ground_truth2 = np.concatenate(total_ground_truth2, axis=0)
            total_ground_truth3 = np.concatenate(total_ground_truth3, axis=0)
            total_pred1 = np.concatenate(total_pred1, axis=0)
            total_pred2 = np.concatenate(total_pred2, axis=0)
            total_pred3 = np.concatenate(total_pred3, axis=0)
            acc1 = accuracy_score(total_ground_truth1, total_pred1)
            acc2 = accuracy_score(total_ground_truth2, total_pred2)
            auc1 = roc_auc_score(total_ground_truth1, total_pred1, multi_class='ovr')
            auc2 = roc_auc_score(total_ground_truth2, total_pred2, multi_class='ovr')
            pre1 = precision_score(total_ground_truth1, total_pred1)
            pre2 = precision_score(total_ground_truth2, total_pred2)
            rec1 = recall_score(total_ground_truth1, total_pred1)
            rec2 = recall_score(total_ground_truth2, total_pred2)
            f1 = f1_score(total_ground_truth1, total_pred1)
            f2 = f1_score(total_ground_truth2, total_pred2)

            r2_score_1 = r2_score(total_ground_truth3, total_pred3, multioutput='raw_values')
            mse_score_1 = mean_squared_error(total_ground_truth3, total_pred3, multioutput='raw_values')
            mae_score_1 = mean_absolute_error(total_ground_truth3, total_pred3, multioutput='raw_values')
            a = 1
            if acc1 >= best_acc1:
                best_acc1 = acc1
                best_pre1 = pre1
                best_rec1 = rec1
                best_f1 = f1
                cm1 = confusion_matrix(total_ground_truth1, total_pred1, labels=[0, 1])
                print(f'Epoch {epoch}, best accuracy of classification task 1 is {best_acc1}.')

            if auc1 > best_auc1:
                best_auc1 = auc1
                print(f'Epoch {epoch}, best auc of classification task 1 is {best_auc1}.')

            if auc2 > best_auc2:
                best_auc2 = auc2
                best_pre2 = pre2
                best_rec2 = rec2
                best_f2 = f2

                print(f'Epoch {epoch}, best auc of classification task 1 is {best_auc2}.')

            if acc2 >= best_acc2:
                best_acc2 = acc2
                cm2 = confusion_matrix(total_ground_truth2, total_pred2, labels=[0, 1])
                print(f'Epoch {epoch}, best accuracy of classification task 2 is {best_acc2}.')
                torch.save(model.state_dict(), 'checkpoints/classification_model.pt')

            if np.sum(r2_score_1) >= scorebest_r2_score:
                scorebest_r2_score = np.sum(r2_score_1)
                best_r2_score_list = r2_score_1
                best_mse = mse_score_1
                best_mae = mae_score_1

                print(f'Epoch {epoch}, best r2_score of regression task 1 is {scorebest_r2_score}.')
                print(f'Epoch {epoch}, best r2_score list of regression task 1 is {best_r2_score_list}.')

    print('###########' * 5)
    print(f'The best accuracy of classification task 1 is {best_acc1}.')
    print(f'The best auc of classification task 1 is {best_auc1}.')
    print(f'The best pre of classification task 1 is {best_pre1}.')
    print(f'The best rec of classification task 1 is {best_rec1}.')
    print(f'The best f1 of classification task 1 is {best_f1}.')
    print(f'The best accuracy of classification task 2 is {best_acc2}.')
    print(f'The best auc of classification task 2 is {best_auc2}.')
    print(f'The best pre of classification task 2 is {best_pre2}.')
    print(f'The best rec of classification task 2 is {best_rec2}.')
    print(f'The best f1 of classification task 2 is {best_f2}.')

    print(f'The r2 score of regression task 1 is {best_r2_score_list}.')
    print(f'The mse score of regression task 1 is {best_mse}.')
    print(f'The mae score of regression task 1 is {best_mae}.')

    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(cm1, annot=True, ax=ax)
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()

    # ----
    f, ax = plt.subplots()
    sns.heatmap(cm2, annot=True, ax=ax)
    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/data.xlsx', type=str, help='')
    parser.add_argument('--train_ratio', default=0.7, type=float, help='the ratio of train set')
    parser.add_argument('--epochs', default=2000, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()

    train(args)
