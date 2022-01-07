import torch as tc
from torch.utils.data import DataLoader, Dataset
import numpy as np

from scipy.spatial.distance import cdist, euclidean


class Dataset_train(Dataset):
    def __init__(self, data, interval =(0.01, 0.99)):
        self.nsamples, self.nfeatures = data.shape
        self.data = data
        self.l = data.shape[0]
        self.interval = interval


    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        p = np.random.uniform(self.interval[0], self.interval[1])

        full_data = self.data[idx, :]
        mask = (tc.rand_like(full_data)<p)*1.0

        x_1, x_2 = full_data.clone(), 1-full_data.clone()


        x_1[mask==0], x_2[mask==0] = 0, 0
        return tc.cat((x_1, x_2), axis = 0), mask, full_data

class Dataset_test(Dataset):
    def __init__(self, data):
        self.nsamples, self.nfeatures = data.shape
        self.data = data
        self.l = data.shape[0]

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        p = 0.5

        full_data = self.data[idx, :]
        mask = (tc.rand_like(full_data)<p)*1.0

        x_1, x_2 = full_data.clone(), 1-full_data.clone()


        x_1[mask==0], x_2[mask==0] = 0, 0
        return tc.cat((x_1, x_2), axis = 0), mask, full_data


class Dataset_train_from_pandas(Dataset):
    def __init__(self, df, interval =(0.01, 0.99)):
        self.nsamples, self.nfeatures = df.shape
        self.df = df
        self.sample_names, self.feature_names, self.data_tensor = np.array(df.index), np.array(df.columns), tc.tensor(np.array(df)).float()
        self.interval = interval


    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        p = np.random.uniform(self.interval[0], self.interval[1])

        full_data = self.data_tensor[idx, :]
        mask = (tc.rand_like(full_data)<p)*1.0

        x_1, x_2 = full_data.clone(), 1-full_data.clone()


        x_1[mask==0], x_2[mask==0] = 0, 0
        return tc.cat((x_1, x_2), axis = 0), mask, full_data


class Dataset_LRP_from_pandas(Dataset):
    def __init__(self, df, target_id, sample_id, maskspersample=10000):
        self.nsamples, self.nfeatures = df.shape
        self.df = df
        self.sample_names, self.feature_names, self.data_tensor = np.array(df.index), np.array(df.columns), tc.tensor(np.array(df)).float()
        self.target_id = target_id
        self.sample_id =sample_id
        self.maskspersample = maskspersample
        


    def __len__(self):
        return self.maskspersample

    def __getitem__(self, idx):

        full_data = self.data_tensor[self.sample_id, :]
        mask = (tc.rand_like(full_data) < 0.5) * 1.0

        mask[self.target_id] = 0

        x_1, x_2 = full_data.clone(), 1-full_data.clone()
        x_1[mask==0], x_2[mask==0] = 0,0
        return tc.cat((x_1, x_2), axis = 0), mask, full_data


