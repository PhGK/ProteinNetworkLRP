from LRPclass_precise import LRP

from dataloading import Dataset_train
from data import load_data_from_frame_overlap
import torch as tc
import sys
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

hidden_factor = 10
hidden_depth = 3
dropout = 0.0
gamma = 0.1
lr = 0.05
nepochs=40001

njobs=10
cuda=False
PATH = '.'

model_path = PATH + '/results/LRP/models/'
RESULTPATH = PATH + '/results/LRP/'


dataframe = pd.read_csv('./data/tcpa_data_051017.csv', index_col = 0)
Reactome_genes = pd.read_csv('./data/int_react_147_060418.csv', delimiter='\t', index_col=0, header=None)
df_filtered = dataframe[list(Reactome_genes.index)]


def calc_all_patients(fold):
    train_data, test_data = load_data_from_frame_overlap(df_filtered)
    print(train_data.shape, test_data.shape)

    model = LRP(train_data.shape[1] * 2, train_data.shape[1], hidden=(train_data.shape[1]) * hidden_factor,
               hidden_depth=hidden_depth, gamma=gamma, dropout=dropout)


    loss = model.train(train_data, test_data, epochs=nepochs, lr=lr)
    print(loss)


    #for sample_id, sample_name in enumerate(test_data.index):
    #    model.compute_network(test_data,sample_name, sample_id,RESULTPATH,device = tc.device("cuda:0" if cuda else "cpu") )

    print('computing LRP...')
    Parallel(n_jobs=njobs)(delayed(model.compute_network)(test_data, sample_name, sample_id, RESULTPATH, device = tc.device("cuda:0" if cuda else "cpu")) 
        for sample_id, sample_name in enumerate(test_data.index))



calc_all_patients(0)
