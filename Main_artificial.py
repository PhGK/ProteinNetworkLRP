from LRPclass_precise import LRP

from dataloading import Dataset_train
from data_artificial import load_data_cv_from_frame, generate_data_homogeneous, generate_data_heterogeneous
import torch as tc
import sys
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

hidden_factor = 10
hidden_depth = 2
dropout = 0.0
gamma = 0.01
nepochs = 3600
lr = 0.03
batch_size = 250

njobs=1
cuda=True
PATH = '.'

datatype = 'homogeneous'
RESULTPATH = PATH + '/results/artificial/' + datatype + '/'
nsamples=4000

if datatype == 'homogeneous':
    df = generate_data_homogeneous(datatype, nsamples)

elif datatype == 'heterogeneous':
    df = generate_data_heterogeneous(datatype, nsamples)

else:
    print('data not found')


if not os.path.exists(RESULTPATH):
    os.makedirs(RESULTPATH)

def calc_all_patients(fold):
    train_data, test_data = load_data_cv_from_frame(df, fold,2)
    print(train_data.shape, test_data.shape)
    if datatype == 'heterogeneous':
            train_data.to_csv('./results/artificial/artificial_heterogeneous_train.csv')
            test_data.to_csv('./results/artificial/artificial_heterogeneous_test.csv')

    model = LRP(train_data.shape[1] * 2, train_data.shape[1], hidden=(train_data.shape[1]) * hidden_factor,
               hidden_depth=hidden_depth, gamma=gamma, dropout=dropout)


    loss = model.train(train_data, test_data, epochs = nepochs, lr=lr, batch_size = batch_size, device = tc.device("cuda:1" if cuda else "cpu"))
    print(loss)


    for sample_id, sample_name in enumerate(test_data.index):
        model.compute_network(test_data,sample_name, sample_id,RESULTPATH,device = tc.device("cuda:1" if cuda else "cpu") )

    #Parallel(n_jobs=njobs)(delayed(model.compute_network)(test_data, sample_name, sample_id, RESULTPATH, device = tc.device("cuda:0" if cuda else "cpu"))
    #    for sample_id, sample_name in enumerate(test_data.index))



calc_all_patients(0)
