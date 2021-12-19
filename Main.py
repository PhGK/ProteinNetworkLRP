from LRPclass import LRP

from dataloading import Dataset_train
from data import load_data_cv_from_frame
import torch as tc
import sys
import os
import pandas as pd
import numpy as np

hidden_factor = 5
hidden_depth = 5
dropout = 0.0
gamma = 0.01

PATH = '.'

model_path = PATH + '/results/LRP/models/'
RESULTPATH = PATH + '/results/LRP/'


dataframe = pd.read_csv('./data/tcpa_data_051017.csv', index_col = 0)
Reactome_genes = pd.read_csv('./data/int_react_147_060418.csv', delimiter='\t', index_col=0, header=None)
df_filtered = dataframe[list(Reactome_genes.index) ]


train_data, test_data = load_data_cv_from_frame(df_filtered, 1,10)

model = LRP(train_data.shape[1] * 2, train_data.shape[1], hidden=(train_data.shape[1]) * hidden_factor,
               hidden_depth=hidden_depth, gamma=gamma, dropout=dropout)


loss = model.train(train_data, test_data, 10)
print(loss)
g
trainloss, testloss = train(model, train_data, test_data, epochs=nepochs, lr=learning_rate, noise_level=noise_factor, weight_decay=wd_factor, batch_size=batch_size,
      device=tc.device("cpu"))

if not os.path.exists(model_path):
    os.makedirs(model_path)
model.cpu()
tc.save(model.cpu(), model_path + data_type + '.pt')

