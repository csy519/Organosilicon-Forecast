# -*- coding: utf-8 -*-

from torch.utils.data import Dataset


class RegressionDataset(Dataset):
    def __init__(self, inputs, labels):
        super(RegressionDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.inputs)


class MultiTaskDataset(Dataset):
    def __init__(self, inputs, labels):
        super(MultiTaskDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]

        return x, y

    def __len__(self):
        return len(self.inputs)