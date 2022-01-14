from LRPmodel import train, Model

from dataloading import Dataset_train
from data import load_data_cv_overlap
import torch as tc
import sys
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

PATH = '.'



model_path = PATH + '/results/crossvalidation/models/'
RESULTPATH = PATH + '/results/crossvalidation/cv.csv'
if os.path.exists(RESULTPATH):
    os.remove(RESULTPATH)

nepochs = 1201
njobs = 10

def crossval(loop):
    for learning_rate in [0.3, 0.1, 0.03, 0.01]:
        for hidden_depth in [2,3,4,5]:
            for hidden_factor in [5,10]:
                print(learning_rate, hidden_depth, hidden_factor)
                train_data, test_data, featurenames, train_names, test_names = load_data_cv_overlap(loop,10)
                model = Model(train_data.shape[1] * 2, train_data.shape[1], hidden=(train_data.shape[1]) * hidden_factor,
                    hidden_depth=hidden_depth)

                losses = train(model, train_data, test_data, epochs=nepochs, lr=learning_rate, batch_size=25,
                    device=tc.device("cpu"))

                losses[['lr', 'depth', 'neurons', 'loop']] = learning_rate, hidden_depth, hidden_factor, loop
                

                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                model.cpu()


                tc.save(model.cpu(), model_path + '_' + str(nepochs) + '_' + str(learning_rate) + '_' + str(hidden_depth) + '_' + 
                    str(hidden_factor) +   '.pt')

                losses.to_csv(RESULTPATH, mode='a', header=not os.path.exists(RESULTPATH))


Parallel(n_jobs=njobs)(delayed(crossval)(loop) for loop in range(10))

