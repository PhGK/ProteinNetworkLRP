from LRPmodel_validation import train, calc_all_paths, Model, createLRPau, createLRPau_withmean

from dataloading_simple import Dataset_train, Dataset_LRP, geometric_median
from data import load_data
import torch as tc
import sys
from joblib import Parallel, delayed
import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve

data_type = sys.argv[1]
gamma = float(sys.argv[2])
hidden_factor = int(sys.argv[3])
noise_factor = float(sys.argv[4])
wd_factor= float(sys.argv[5])
learning_rate= float(sys.argv[6])
PATH = sys.argv[7]
hidden_depth = int(sys.argv[8])
nepochs = int(sys.argv[9])
normalize = sys.argv[10] == 'norm'
njobs = int(sys.argv[11])
dropout = float(sys.argv[12])
batch_size = int(sys.argv[13])
traintest = sys.argv[14]
datadropout = sys.argv[15] == 'dropout'

model_path = PATH + '/results/' + data_type + '/models/'
result_path = PATH + '/results/' + data_type + '/LRP_values/'
au_path = PATH + '/results/' + data_type + '/LRP_au/'
AUC_PATH = PATH + '/results/ValidationResults.csv'
train_data, test_data, featurenames, train_names, test_names, ground_truth = load_data(data_type, normalize= normalize, double=False, dropout = datadropout)

print(len(train_names), len(test_names))
#########################################
print('training model', train_data.shape)
#print('actual hidden depth:',hidden_depth*(train_data.shape[1]//8))

model = Model(train_data.shape[1] * 2, train_data.shape[1], hidden=(train_data.shape[1]) * hidden_factor,
               hidden_depth=hidden_depth, gamma=gamma, dropout=dropout)
trainloss, testloss = train(model, train_data, test_data, epochs=nepochs, lr=learning_rate, noise_level=noise_factor, weight_decay=wd_factor, batch_size=batch_size,
      device=tc.device("cpu"))

if not os.path.exists(model_path):
    os.makedirs(model_path)
model.cpu()
tc.save(model.cpu(), model_path + data_type + '.pt')


################################################

print('loading old model')
#model = tc.load(model_path + data_type + '.pt', map_location=tc.device('cpu'))
print('training finished, starting LRP')

if not os.path.exists(result_path + 'raw_data/'):
    os.makedirs(result_path + 'raw_data/')
files = os.listdir(result_path + 'raw_data/')
for fil in files:
    os.remove(result_path + 'raw_data/' + fil)


if traintest == 'traindata':
    print('traindata is used')
    Parallel(n_jobs=njobs)(delayed(calc_all_paths)(model, train_data, id,name, featurenames = featurenames, data_type = data_type, result_path = result_path,
                                               device = tc.device("cpu")) for id, name in enumerate(train_names))
else:
    Parallel(n_jobs=njobs)(delayed(calc_all_paths)(model, test_data, id,name, featurenames = featurenames, data_type = data_type, result_path = result_path,
                                               device = tc.device("cpu")) for id, name in enumerate(test_names))
    print('testdata is used')
    


###################
print('raw LRP finished, starting LRP au')

filenames = os.listdir(result_path + 'raw_data/')
all_files = pd.concat([pd.read_csv(result_path + 'raw_data/' + filename) for filename in filenames],axis=0)
LRPau = createLRPau(all_files)
LRPau_withmean = createLRPau_withmean(all_files)
meanLRPau, meanLRPau_withmean = LRPau.groupby(['source_gene', 'target_gene']).mean().reset_index(), LRPau_withmean.groupby(['source_gene', 'target_gene']).mean().reset_index()

##
meanLRPau = meanLRPau.merge(ground_truth, how='left',  on = ['source_gene', 'target_gene'])
meanLRPau['gt'][np.isnan(meanLRPau['gt'])] = 0
meanLRPau['gt'] = meanLRPau['gt'].astype(int)
print(meanLRPau['gt'], meanLRPau['LRP'])
fpr, tpr, thresholds = roc_curve(meanLRPau['gt'], meanLRPau['LRP'])
meanAUC = auc(fpr, tpr)
##
meanLRPau_withmean = meanLRPau_withmean.merge(ground_truth, how='left',  on = ['source_gene', 'target_gene'])
meanLRPau_withmean['gt'][np.isnan(meanLRPau_withmean['gt'])] = 0
meanLRPau_withmean['gt'] = meanLRPau_withmean['gt'].astype(int)
fpr, tpr, thresholds = roc_curve(meanLRPau_withmean['gt'], meanLRPau_withmean['LRP'])
meanAUC_withmean = auc(fpr, tpr)
##
meanLRPau['mtm'] = meanLRPau['LRP']+meanLRPau_withmean['LRP']
fpr, tpr, thresholds = roc_curve(meanLRPau['gt'], meanLRPau['mtm'])
mtmAUC = auc(fpr, tpr)
##
#build geometric median
LRPau_interaction = LRPau_withmean.copy()
LRPau_interaction['interaction'] = LRPau_withmean['source_gene'] + '_' + LRPau_withmean['target_gene']
LRPau_interaction = LRPau_interaction[['sample_name', 'interaction', 'LRP']]

LRPau_wide = LRPau_interaction.pivot(index = ['sample_name'],columns=['interaction'], values = 'LRP')
geom_median = pd.DataFrame({'LRP': geometric_median(np.array(LRPau_wide))})
geom_median['interaction'] = LRPau_wide.columns

geom_interactions = pd.DataFrame(geom_median.interaction.str.split('_').tolist(), columns = ['source_gene', 'target_gene'])
geom_median = pd.concat((geom_median, geom_interactions), axis=1)
geom_median = geom_median.merge(ground_truth, how='left',  on = ['source_gene', 'target_gene'])
geom_median['gt'][np.isnan(geom_median['gt'])] = 0
geom_median['gt'] = geom_median['gt'].astype(int)
fpr, tpr, thresholds = roc_curve(geom_median['gt'], geom_median['LRP'])
geom_medianAUC = auc(fpr, tpr)


##
results = pd.DataFrame({'data_type': data_type, 'gamma': gamma, 'hidden_factor': hidden_factor, 'lr': learning_rate, 'dropout': dropout, 'datadropout':datadropout,
                        'weight_decay': wd_factor, 'noise': noise_factor, 'normalize': normalize,'hidden_depth': hidden_depth, 'nepochs': nepochs, 'batch_size': batch_size,
                        'traintest': traintest,
                        'meanAUC': meanAUC,'meanAUC_withmean': meanAUC_withmean, 'mtmAUC': mtmAUC, 'geom_medianAUC': geom_medianAUC,'network_size': train_data.size(1),'trainloss': trainloss.item(),
                        'testloss': testloss.item()}, index=[0])


results.to_csv(AUC_PATH, mode='a', header=not os.path.exists(AUC_PATH))

