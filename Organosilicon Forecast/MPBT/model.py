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
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle


class RegressionModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(RegressionModel, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.layer1 = nn.Linear(14, 200)
        self.layer2 = nn.Linear(200, 300)
        self.layer3 = nn.Linear(300, 300)
        # self.layer4 = nn.Linear(200, 500)
        self.layer5 = nn.Linear(300, 100)
        self.layer6 = nn.Linear(100, 2)
    
    def forward(self, x):
        out = F.leaky_relu(self.layer1(x))
        out = self.dropout1(out)
        out = F.leaky_relu(self.layer2(out))
        out = self.dropout2(out)
        out = F.leaky_relu(self.layer3(out))
        out = self.dropout3(out)
        # out = F.leaky_relu(self.layer4(out))
        # out = self.dropout4(out)
        out = F.leaky_relu(self.layer5(out))
        out = self.dropout5(out)
        out = self.layer6(out)

        return out


class MultiTaskModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(MultiTaskModel, self).__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.dropout7 = nn.Dropout(dropout)
        self.dropout8 = nn.Dropout(dropout)
        self.dropout9 = nn.Dropout(dropout)

        self.layer1 = nn.Linear(16, 200)
        self.layer2 = nn.Linear(200, 300)
        # self.layer3 = nn.Linear(300, 200)

        # branch 2: regression task 1
        # self.branch2_1 = nn.Linear(300, 500)
        # self.branch2_2 = nn.Linear(500, 300)
        # self.branch2_3 = nn.Linear(300, 200)
        # self.branch2_4 = nn.Linear(200, 3)

        self.branch2_1 = nn.Linear(300, 800)
        self.branch2_2 = nn.Linear(800, 1000)
        self.branch2_3 = nn.Linear(1000, 300)
        self.branch2_4 = nn.Linear(300, 3)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.leaky_relu(self.layer2(x))
        x = self.dropout2(x)
        # x = F.leaky_relu(self.layer3(x))
        # x = self.dropout3(x)
    
        output2 = F.leaky_relu(self.branch2_1(x))
        output2 = self.dropout7(output2)
        output2 = F.leaky_relu(self.branch2_2(output2))
        output2 = self.dropout8(output2)
        output2 = F.leaky_relu(self.branch2_3(output2))
        output2 = self.dropout9(output2)
        output2 = self.branch2_4(output2)

        return output2


# class MultiTaskModel(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(MultiTaskModel, self).__init__()
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#         self.dropout4 = nn.Dropout(dropout)
#         self.dropout5 = nn.Dropout(dropout)
#
#         self.layer1 = nn.Linear(17, 200)
#         self.layer2 = nn.Linear(200, 300)
#         self.layer3 = nn.Linear(300, 300)
#
#         # Branch 1
#         self.branch1_1 = nn.Linear(200, 500)
#         self.branch1_2 = nn.Linear(300, 100)
#         self.branch1_3 = nn.Linear(100, 2)
#
#         # Branch 2
#         self.branch2_1 = nn.Linear(200, 500)
#         self.branch2_2 = nn.Linear(300, 100)
#         self.branch2_3 = nn.Linear(100, 2)
#
#     def forward(self, x):
#         out = F.leaky_relu(self.layer1(x))
#         out = self.dropout1(out)
#         out = F.leaky_relu(self.layer2(out))
#         out = self.dropout2(out)
#         out = F.leaky_relu(self.layer3(out))
#         out = self.dropout3(out)
#
#         branch1_out =
#         # out = F.leaky_relu(self.layer4(out))
#         # out = self.dropout4(out)
#         out = F.leaky_relu(self.layer5(out))
#         out = self.dropout5(out)
#         out = self.layer6(out)
#
#         return out