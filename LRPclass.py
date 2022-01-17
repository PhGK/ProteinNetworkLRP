import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from torch.utils.data import Dataset, DataLoader

from itertools import permutations
import pandas as pd
import os

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


from itertools import permutations
import pandas as pd
from dataloading import Dataset_train_from_pandas, Dataset_LRP_from_pandas
import os



class LRP_Linear(nn.Module):
    def __init__(self, inp, outp, gamma=0.01, eps=0.0):
        super(LRP_Linear, self).__init__()
        self.A_dict = {}
        self.linear = nn.Linear(inp, outp)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_normal_(self.linear.weight)
        #nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        self.gamma = tc.tensor(gamma)
        self.eps = tc.tensor(eps)
        self.rho = None
        self.iteration = None

    def forward(self, x):

        if not self.training:
            self.A_dict[self.iteration] = x.clone()
        return self.linear(x)

    def relprop(self, R):
        device = next(self.parameters()).device
        # print('rel', self.iteration, self.A_dict[self.iteration].sum())
        A = self.A_dict[self.iteration].clone()
        A, self.eps = A.to(device), self.eps.to(device)
        Ap = tc.where(A > self.eps, A, tc.tensor(1e-9).to(device)).detach().data.requires_grad_(True)
        Am = tc.where(A < -self.eps, A, tc.tensor(-1e-9).to(device)).detach().data.requires_grad_(True)

        zpp = self.newlayer(1).forward(Ap)  # + self.eps
        zmm = self.newlayer(-1, no_bias=True).forward(Am)  # + self.eps

        zmp = self.newlayer(1, no_bias=True).forward(Am)  # - self.eps
        zpm = self.newlayer(-1).forward(Ap)  # - self.eps

        with tc.no_grad():
            Y = self.forward(A).data
        sp = ((Y > self.eps).float() * R / (zpp + zmm + self.eps)).data  # .requires_grad_(True)
        sm = ((Y < -self.eps).float()* R / (zmp + zpm + self.eps)).data  # .requires_grad_(True)

        (zpp * sp).sum().backward()
        cpp = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zpm * sm).sum().backward()
        cpm = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zmp * sp).sum().backward()
        cmp = Am.grad
        # Am.grad.detach_()
        # Am.grad.zero_()
        Am.grad = None
        Am.requires_grad_(True)

        (zmm * sm).sum().backward()
        cmm = Am.grad
        # Am.grad.detach_()
        # Am.grad.zero_()
        Am.grad = None
        Am.requires_grad_(True)


        R_1 = (Ap * cpp).data
        R_2 = (Ap * cpm).data
        R_3 = (Am * cmp).data
        R_4 = (Am * cmm).data
        # print('R1',R_1,R_2,'R3', R_3,R_4)
        
        return R_1 + R_2 + R_3 + R_4

    def newlayer(self, sign, no_bias=False):

        if sign == 1:
            rho = lambda p: p + self.gamma * p.clamp(min=1e-9)
        else:
            rho = lambda p: p - self.gamma * p.clamp(max=-1e-9)

        layer_new = copy.deepcopy(self.linear)

        try:
            layer_new.weight = nn.Parameter(rho(self.linear.weight))
        except AttributeError:
            pass

        try:
            layer_new.bias = nn.Parameter(self.linear.bias * 0 if no_bias else rho(self.linear.bias))
        except AttributeError:
            pass


        return layer_new


class LRP_ReLU(nn.Module):
    def __init__(self):
        super(LRP_ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def relprop(self, R):
        return R


class LRP_DropOut(nn.Module):
    def __init__(self, p):
        super(LRP_DropOut, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

    def relprop(self, R):
        return R

class LRP_noise(nn.Module):
    def __init__(self, p):
        super(LRP_noise, self).__init__()
        self.p = p
    def forward(self, x):
        if self.train:
            x += self.p*tc.randn_like(x)
        return x

    def relprop(self, R):
        return R


class Model(nn.Module):
    def __init__(self, inp, outp, hidden, hidden_depth, gamma=0.01, dropout = 0.0):
        super(Model, self).__init__()
        self.layers = nn.Sequential(LRP_Linear(inp, hidden, gamma=gamma), LRP_ReLU(), LRP_DropOut(dropout))
        for i in range(hidden_depth):
            self.layers.add_module('LRP_Linear' + str(i + 1), LRP_Linear(hidden, hidden, gamma=gamma))
            self.layers.add_module('LRP_ReLU' + str(i + 1), LRP_ReLU())
            self.layers.add_module('LRP_DropOut' + str(i + 1), LRP_DropOut(dropout))
        self.layers.add_module('LRP_Linear_last', LRP_Linear(hidden, outp, gamma=gamma))

        if hidden_depth==-1:
            self.layers = nn.Sequential(LRP_Linear2(inp, outp, gamma=gamma))

    def forward(self, x):
        return self.layers.forward(x)

    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R


class LRP:
    def __init__(self, inp, outp, hidden, hidden_depth, gamma=0.01, dropout = 0.0):
        
        self.model = Model(inp, outp, hidden=hidden,
               hidden_depth=hidden_depth, gamma=gamma, dropout=dropout)

    def train(self, train_data, test_data, epochs, lr = 0.01, batch_size=25, device=tc.device('cpu')):
        nsamples, nfeatures = train_data.shape
        optimizer = tc.optim.SGD(self.model.parameters(), lr=lr)#, momentum=0.9)
        #optimizer = tc.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.model.train().to(device)

        losses = []
        for epoch in range(epochs):
            trainset = Dataset_train_from_pandas(train_data)
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

            for masked_data, mask, full_data in trainloader:
                optimizer.zero_grad()
                masked_data = masked_data.to(device)
                mask = mask.to(device)
                full_data = full_data.to(device)

                pred = self.model(masked_data)
                loss = criterion(pred[mask==0], full_data[mask==0]) 
                loss.backward()
                optimizer.step()

	    

            if epoch in [1,5,10,20,50, 100, 150,200, 250, 300, 350, 400, 600,800,1000,1200,5000,10000,20000,30000,40000]:
                print(epoch)
                self.model.eval()
                testset = Dataset_train_from_pandas(test_data)
                testloader = DataLoader(testset, batch_size=test_data.shape[0], shuffle=False)

                for masked_data, mask, full_data in testloader:
                    masked_data = masked_data.to(device)
                    mask = mask.to(device)
                    full_data = full_data.to(device)
                    with tc.no_grad():
                        pred = self.model(masked_data)
                    testloss = criterion(pred[mask==0], full_data[mask==0])
                    break

                losses.append(pd.DataFrame({'trainloss': [loss.detach().cpu().numpy()], 'testloss': [testloss.cpu().numpy()], 'epoch': [epoch]}))

        return pd.concat(losses)


    def compute_LRP(self, test_set, target_id, sample_id, batch_size, device = tc.device('cpu')):

        criterion = nn.MSELoss()
        testset = Dataset_LRP_from_pandas(test_set, target_id, sample_id)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        self.model.to(device).eval()

        masked_data, mask, full_data = next(iter(testloader))

        masked_data, mask, full_data = masked_data.to(device), mask.to(device), full_data.to(device)

        pred = self.model(masked_data)

        error = criterion(pred.detach()[:,target_id], full_data.detach()[:,target_id]).cpu().numpy()
        y = full_data.detach()[:,target_id].cpu().mean().numpy()
        y_pred = pred.detach()[:,target_id].cpu().mean().numpy()

        R = tc.zeros_like(pred)
        R[:,target_id] = pred[:,target_id].clone()

        a = self.model.relprop(R)

        LRP_sum = (a.mean(dim=0))

        LRP_unexpanded = 0.5 * (LRP_sum[:LRP_sum.shape[0] // 2] + LRP_sum[LRP_sum.shape[0] // 2:])

        #mask_sum = mask.sum(dim=0).float()

        LRP_scaled = LRP_unexpanded#/mask_sum
        #print(LRP_scaled)
        #LRP_scaled = tc.where(tc.isnan(LRP_scaled),tc.tensor(0.0).to(device), LRP_scaled)

        return LRP_scaled.cpu().numpy(), error, y , y_pred


    def compute_network(self, test_data, sample_name, sample_id, result_path, device = tc.device('cuda:0'), run = 0):
        end_frame = []

        sample_names, feature_names = np.array(test_data.index), np.array(test_data.columns)
        for target in range(test_data.shape[1]):
            LRP_value, error, y, y_pred = self.compute_LRP(test_data, target, sample_id, batch_size = 100, device = device)
            assert sample_name == sample_names[sample_id], 'sample names not correct'

            frame = pd.DataFrame({'LRP': LRP_value, 'predicting_protein': feature_names, 'masked_protein': feature_names[target] ,'sample_name': sample_name, 'error':error, 'y':y, 'y_pred':y_pred})
            end_frame.append(frame)
            if any(abs(LRP_value)>10):
                print(frame)
            end_result_path = result_path + 'raw_data/' + 'LRP_' + str(sample_id) + '_' + str(sample_name) +'.csv'
            if not os.path.exists(result_path + 'raw_data/'):
                os.makedirs(result_path + 'raw_data/')

 
        end_frame = pd.concat(end_frame, axis=0)
        end_frame['sample_name'] = sample_name
        end_frame.to_csv(end_result_path)


def createLRPau_withmean(data):
    data['LRP'] = np.abs(data['LRP'])
    sym_dir = data[['source_gene', 'target_gene', 'sample_name', 'LRP', 'y_pred', 'y']]
    sym_trans = sym_dir.copy()
    sym_trans.columns = ['target_gene', 'source_gene', 'sample_name', 'LRP', 'ty_pred', 'ty']

    sym = sym_dir.merge(sym_trans, on=['source_gene', 'target_gene', 'sample_name'])

    sym['LRP'] = sym[['LRP_x', 'LRP_y']].mean(axis=1)#sym.apply(lambda row: np.mean(row.LRP_x, row.LRP_y), axis=1)
    #sym['LRP'] = np.argsort(np.abs(sym['LRP']))
    output = sym[sym['source_gene'] > sym['target_gene']]


    result = output[['LRP', 'source_gene', 'target_gene', 'sample_name']]
    return result
