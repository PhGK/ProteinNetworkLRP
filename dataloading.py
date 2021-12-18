import torch as tc
from torch.utils.data import DataLoader, Dataset
import numpy as np

from scipy.spatial.distance import cdist, euclidean

def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


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

        x_1, x_2 = full_data, 1-full_data


        x_1[mask==0], x_2[mask==0] = 0, 0
        return tc.cat((x_1, x_2), axis = 0), mask, full_data



class Dataset_LRP(Dataset):
    def __init__(self, data, target_id, sample_id, maskspersample=10000, interval =(0.01, 0.99)):
        self.nsamples, self.nfeatures = data.shape
        self.data = data
        self.l = data.shape[0]
        self.target_id = target_id
        self.sample_id =sample_id
        self.maskspersample = maskspersample
        self.interval = interval

    def __len__(self):
        return self.maskspersample

    def __getitem__(self, idx):
        #p = np.random.uniform(self.interval[0], self.interval[1])
        full_data = self.data[self.sample_id, :]
        mask = (tc.rand_like(full_data) < 0.5) * 1.0

        mask[self.target_id] = 0
        noise = tc.zeros_like(full_data)
        #noise = 0.1 * tc.randn_like(full_data)
        x_1, x_2 = full_data.clone() + noise, 1-full_data.clone() - noise
        x_1[mask==0], x_2[mask==0] = 0,0
        return tc.cat((x_1, x_2), axis = 0), mask, full_data


class Dataset_LRP_mean(Dataset):
    def __init__(self, data, target_id, sample_id, maskspersample=10000, interval =(0.01, 0.99)):
        self.nsamples, self.nfeatures = data.shape
        #self.data = data.median(axis=0, keepdim =True)[0].repeat(self.nsamples,1) #+ 0.1*tc.randn_like(data)
        self.data = tc.tensor(np.array(self.nsamples*[geometric_median(np.array(data))])).float()
        #print(data.shape, self.data.shape)
        self.l = data.shape[0]
        self.target_id = target_id
        self.sample_id =sample_id
        self.maskspersample = maskspersample
        self.interval = interval

    def __len__(self):
        return self.maskspersample

    def __getitem__(self, idx):
        #p = np.random.uniform(self.interval[0], self.interval[1])
        full_data = self.data[self.sample_id, :]
        mask = (tc.rand_like(full_data) < 0.9) * 1.0

        mask[self.target_id] = 0
        noise = tc.zeros_like(full_data)
        #noise = 0.1 * tc.randn_like(full_data)
        x_1, x_2 = full_data.clone() + noise, 1-full_data.clone() - noise
        x_1[mask==0], x_2[mask==0] = 0,0
        return tc.cat((x_1, x_2), axis = 0), mask, full_data


