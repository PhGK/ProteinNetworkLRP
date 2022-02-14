# ProteinNetworkLRP
Predicting patient-level proteomic networks by explainable artificial intelligence

This repository contains all data and code related to our manuscript 'Predicting patient-level proteomic networks by explainable artificial intelligence'. It allows the replication of all experiments and provides users with a python class to predict networks for individual samples for their own data.

All experiments that calculate LRP values are based on pytorch. In order to efficiently process large datasets computing can be parallelized and run on the gpu.

**1) Computation of biological networks**  
The computation of protein interaction networks for individual patient's tumors can be replicated by running the script Main.py. The script allows the setting of several hyperparameters.
- hidden_factor: Integer that defines the width of hidden layers (= hidden_factor * number of proteins).
- hidden_depth: number of hidden layers -1
- gamma: LRP hyper parameter
- lr: learning rate (we currently use gradient descent with momentum)
- nbatch: batch size
- cuda: Boolean. If True, experiment will run on the gpu
- njobs: number of parallel jobs. This will not affect the training of the network.
Results can be found in results/LRP/raw_data

**2) Computation of artificial networks**  
The script Main_artificial.py replicates results from Figure 3 of the manuscript. The script works like 1), but users can set datatype to 'homogeneous' or 'heterogeneous' to generate data with either one or multiple different underlying networks.

Results can be found in results/artificial/homogeneous/raw_data or results/artificial/heterogeneous/raw_data

**3) Crossvalidation**  
Model crossvalidation for the biological networks can be replicated using the script 'Crossvalidation.py'. This can be parallelized using njobs.
Results can be found in results/crossvalidation

**4) Customized individual network predictions**  
Using the LRP class, we facilitate the computation of networks for individual samples.
First, an instance of the LRP model is created.

``` 
from LRPclass_precise import LRP

model = LRP(2*nfeatures, nfeatures, hidden_features, network_depth)
```
Note that the input dimension of the model is twice the feature dimension of the original sample. This is due to the feature expansion which is necessary for the imputation task. 
train_data is a pandas dataframe with size n_train_samples * n_features. In order to prevent overfitting, a test data set must be provided with equal amount of features. 

```
cuda=False
model.train(train_data, test_data, epochs, lr = 0.03, batch_size=250, device=tc.device(tc.device("cuda:0" if cuda else "cpu"))
```

 use_data is a pandas frame with size n_test_samples * n_features (Number of features must be equal to number of features of train_data).
compute_network() computes the network of one sample from `use_data`. The sample must be identified by its sample_id (row number) and and additional sample name.
The individual network will be saved in RESULTPATH.

```
model.compute_network(use_data, sample_name, sample_id, RESULTPATH, device = tc.device("cuda:0" if cuda else "cpu")
```

We show a minimum working example in Working example.ipynb
