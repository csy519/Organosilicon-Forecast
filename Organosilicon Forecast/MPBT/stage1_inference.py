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
from model import RegressionModel


class RegressionDataset(Dataset):
    def __init__(self, inputs):
        super(RegressionDataset, self).__init__()
        self.inputs = inputs

    def __getitem__(self, idx):
        x = self.inputs[idx]

        return x

    def __len__(self):
        return len(self.inputs)


def inference(args):
    origin_data = pd.read_excel(args.data_path)

    scaler1 = pickle.load(open(args.scaler1_path, 'rb'))
    scaler2 = pickle.load(open(args.scaler2_path, 'rb'))

    X = scaler1.transform(origin_data.values).astype(np.float32)

    inference_dataset = RegressionDataset(X)
    inference_dataloader = DataLoader(inference_dataset, batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = RegressionModel()
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    if use_cuda:
        model = model.to(device)

    total_pred = []
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(inference_dataloader), desc=f'Inference') as pbar:
            for x in inference_dataloader:
                x = x.to(device)
                o = model(x)

                total_pred.append(o.cpu().numpy())
                pbar.update(1)
    total_pred = np.concatenate(total_pred, axis=0)
    total_pred = scaler2.inverse_transform(total_pred)
    predict_df = pd.DataFrame(total_pred)
    results_df = pd.concat([origin_data, predict_df], axis=1)

    results_df.to_csv(args.output_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/杭州样品筛选.xlsx', type=str, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--ckpt_path', default='checkpoints/regression_model.pt', type=str, help='checkpoint path')
    parser.add_argument('--scaler1_path', default='checkpoints/regression_scaler1.pkl', type=str, help='scaler path')
    parser.add_argument('--scaler2_path', default='checkpoints/regression_scaler2.pkl', type=str, help='scaler path')
    parser.add_argument('--output_path', default='results/样品筛选kowkaw.csv', type=str, help='output path')
    args = parser.parse_args()

    inference(args)



