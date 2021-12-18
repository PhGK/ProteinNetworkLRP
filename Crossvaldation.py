from LRPmodel import train, Model

from dataloading import Dataset_train
from data import load_data_cv
import torch as tc
import sys
import os
import pandas as pd
import numpy as np

hidden_factor = int(sys.argv[3])
learning_rate= float(sys.argv[6])
hidden_depth = int(sys.argv[8])
nepochs = int(sys.argv[9])
PATH = '.'

model_path = PATH + '/results/models/'


train_data, test_data, featurenames, train_names, test_names = load_data_cv(1,10)
model = Model(train_data.shape[1] * 2, train_data.shape[1], hidden=(train_data.shape[1]) * hidden_factor,
               hidden_depth=hidden_depth, gamma=gamma, dropout=dropout)


trainloss, testloss = train(model, train_data, test_data, epochs=nepochs, lr=learning_rate, noise_level=noise_factor, weight_decay=wd_factor, batch_size=batch_size,
      device=tc.device("cpu"))

if not os.path.exists(model_path):
    os.makedirs(model_path)
model.cpu()
tc.save(model.cpu(), model_path + data_type + '.pt')
