import pandas as pd
import torch as tc
import numpy as np
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer

def load_data():

    dataframe = pd.read_csv('./data/tcpa_data_051017.csv')
    sample_name, cancer_type = dataframe['ID'], dataframe['Cancer_Type']
    data = dataframe.loc[:, (dataframe.columns !='ID') & (dataframe.columns != 'Cancer_Type')]


    print(sample_name, cancer_type)
    print(data)

    g



    return train_set.float(), test_set.float(), feature_names, train_names, test_names, ground_truth



def load_data_cv(loop, number_of_loops):
    dataframe = pd.read_csv('./data/tcpa_data_051017.csv')

    Reactome_genes = pd.read_csv('./data/int_react_147_060418.csv', delimiter='\t', index_col=0, header=None)
    df_filtered = dataframe[['ID'] + list(Reactome_genes.index) ]

    print(df_filtered.shape)
    nsamples, nfeatures = df_filtered.shape
    sample_names, feature_names = np.array(df_filtered['ID']), np.array(df_filtered.columns)
    data = df_filtered.loc[:, (df_filtered.columns !='ID') & (df_filtered.columns != 'Cancer_Type')]

    data_tensor = tc.tensor(np.array(data))

    np.random.seed(0)
    indices = np.random.permutation(nsamples)
    chunks = np.array_split(indices, number_of_loops)
   
    test_indices = chunks[loop]
    train_indices = np.delete(indices, test_indices)
    
    test_data = data_tensor[test_indices,:]
    test_names = sample_names[test_indices]

    train_data = data_tensor[train_indices,:]
    train_names = sample_names[train_indices]


    return train_data.float(), test_data.float(), feature_names, train_names, test_names



load_data_cv(1,10)




