import pandas as pd
import torch as tc
import numpy as np
from scipy.linalg import block_diag
import torch as tc
import torch.nn as nn
import scipy.linalg as linalg
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import RobustScaler, PowerTransformer, QuantileTransformer

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

    meanv, sdv = train_data.mean(axis=0), train_data.std(axis=0)

    train_data = (train_data-meanv)/sdv
    test_data = (test_data-meanv)/sdv
    #todo: normalize

    return train_data.float(), test_data.float(), feature_names, train_names, test_names


def load_data_cv_overlap(loop, number_of_loops):
    dataframe = pd.read_csv('./data/tcpa_data_051017.csv')

    Reactome_genes = pd.read_csv('./data/int_react_147_060418.csv', delimiter='\t', index_col=0, header=None)
    df_filtered = dataframe[['ID'] + list(Reactome_genes.index) ]

    print(df_filtered.shape)
    nsamples, nfeatures = df_filtered.shape
    sample_names, feature_names = np.array(df_filtered['ID']), np.array(df_filtered.columns)
    data = df_filtered.loc[:, (df_filtered.columns !='ID') & (df_filtered.columns != 'Cancer_Type')]

    data_tensor = tc.tensor(np.array(data))

    np.random.seed(loop)
    indices = np.random.permutation(nsamples)
   
    test_indices = indices[:int(indices.shape[0]*0.25)]
    train_indices = indices[int(indices.shape[0]*0.25):]
    
    test_data = data_tensor[test_indices,:]
    test_names = sample_names[test_indices]

    train_data = data_tensor[train_indices,:]
    train_names = sample_names[train_indices]

    meanv, sdv = train_data.mean(axis=0), train_data.std(axis=0)

    train_data = (train_data-meanv)/sdv
    test_data = (test_data-meanv)/sdv
    #todo: normalize

    return train_data.float(), test_data.float(), feature_names, train_names, test_names


def load_data_cv_from_frame(df, loop, number_of_loops):

    nsamples, nfeatures = df.shape
    sample_names, feature_names = np.array(df.index), np.array(df.columns)

    np.random.seed(0)
    indices = np.random.permutation(nsamples)
    chunks = np.array_split(indices, number_of_loops)
   
    test_indices = chunks[loop]
    train_indices = np.delete(indices, test_indices)
    
    test_data = df.iloc[test_indices,:]
    train_data = df.iloc[train_indices,:]

    meanv, sdv = train_data.mean(axis=0), train_data.std(axis=0)

    train_data = (train_data-meanv)/sdv
    test_data = (test_data-meanv)/sdv
    print(test_data.mean(axis=0), test_data.std(axis=0))

    return train_data, test_data



class AR():
    def __init__(self, vector_size, graph):
        self.vector_size = vector_size

        tc.manual_seed(0)
        self.init_vector = tc.rand(vector_size)
        self.diagonal = tc.eye(vector_size) * tc.rand(vector_size)
        self.eigenvectors = tc.tensor(linalg.orth(tc.rand(vector_size, vector_size) + 1 * tc.eye(vector_size)))

        self.function = tc.mm(tc.mm(self.eigenvectors * tc.tensor(graph).float(), self.diagonal),
                              tc.inverse(self.eigenvectors)) * tc.tensor(graph).float()

        self.diagonal2 = tc.eye(vector_size) * tc.rand(vector_size)
        self.eigenvectors2 = tc.tensor(linalg.orth(tc.rand(vector_size, vector_size) + 1 * tc.eye(vector_size)))
        self.function2 = tc.mm(tc.mm(self.eigenvectors2 * tc.tensor(graph).float(), self.diagonal2),
                               tc.inverse(self.eigenvectors2)) * tc.tensor(graph).float()

        # print(linalg.eigvals(self.function))
        self.time_series = [(self.init_vector)]
        self.values = None
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()
        # print(self.function)

        self.covariance, self.meanv = make_spd_matrix(vector_size, random_state=None), tc.zeros(vector_size)
        self.dist = tc.distributions.multivariate_normal.MultivariateNormal(self.meanv, tc.tensor(
            self.covariance).float())  # noise distribution

    def step(self, n, correlated=True):
        for i in range(n):
            eps = self.dist.sample() if correlated else 1 * tc.randn(self.vector_size)
            eps2 = self.dist.sample() if correlated else 1 * tc.randn(self.vector_size)

            next_value = tc.matmul(self.function, (self.time_series[-1] + eps))
            next_value = self.activation1(next_value)

        next_value = tc.matmul(self.function2, (next_value))
        next_value = self.activation2(next_value)

        self.time_series.append(next_value)


        self.values = np.array(tc.stack(self.time_series, dim=0))


    def get_set(self, size, warmup=50):
        self.time_series = [(self.init_vector)]
        self.step(warmup + size - 1)
        return (self.values[warmup:, :], np.array(self.function))


    def get_sampled_set(self, size, warmup=50):
        set = []

        for i in range(size):
            self.time_series = [tc.rand(self.vector_size)]
            self.step(warmup)
            set.append(self.time_series[-1])
        #np.savetxt('/mnt/scratch2/mlprot/LRP_experiment/plots_statistics/use_data/time_series_tanh_one_sample.csv',
        #              self.values, delimiter='\t')
        return np.array(tc.stack(set, dim=0)), np.array(self.function)


def generate_data_homogeneous(datatype, nsamples, nfeatures=32, block_size = (8,8), correlated=True):

    def block_generator(connectivity, block_size):
        x = np.random.mtrand.RandomState(0).binomial(size=block_size[0] * block_size[1], n=1, p=connectivity).reshape(
            block_size)
        return x * x.transpose()  # symmetrize --> leads to sparsity**2

    def specific_network(subgraph):
        blocks = [block_generator(1.0,block_size) if i in subgraph else block_generator(0, block_size) for i
                  in range(matrix_size[0] // block_size[0])]
        block_matrix = np.array(block_diag(*blocks))
        np.fill_diagonal(block_matrix, 1)
        return block_matrix


    matrix_size = (nfeatures, nfeatures)
    ############################
    sparse_graph = specific_network([0, 1, 2, 3])
    full_graph = sparse_graph * 0 + 1


    ar_model = AR(nfeatures, specific_network([0, 1, 2, 3]))
    data = ar_model.get_sampled_set(nsamples, 50)[0]
    df = pd.DataFrame(data)
    df.to_csv('./data/artificial_homogeneous.csv')
    return df


def generate_data_heterogeneous(datatype, nsamples, nfeatures=32, block_size=(8, 8), n=4):
    def block_generator(connectivity, block_size):
        x = np.random.mtrand.RandomState(0).binomial(size=block_size[0] * block_size[1], n=1, p=connectivity).reshape(
            block_size)
        return x * x.transpose()  # symmetrize --> leads to sparsity**2

    def specific_network(subgraph):
        blocks = [block_generator(1.0,block_size) if i in subgraph else block_generator(0, block_size) for i
                  in range(matrix_size[0] // block_size[0])]
        block_matrix = np.array(block_diag(*blocks))
        np.fill_diagonal(block_matrix, 1)
        return block_matrix

    matrix_size = (nfeatures, nfeatures)
    ############################
#    sparse_graph = specific_network([0, 1, 2, 3])
#    full_graph = sparse_graph * 0 + 1

    ar_models = [AR(nfeatures, specific_network([i])) for i in range(n)]
    data = np.concatenate([ar.get_sampled_set(nsamples//n,50)[0] for ar in ar_models], axis = 0)
    print(data.shape)
    
    df = pd.DataFrame(data)

    df.to_csv('./data/artificial_heterogeneous.csv')
    return df
