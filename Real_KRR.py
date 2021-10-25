#!/usr/bin/env python
# coding: utf-8

# KRR with lambda optimized using the scikit learn package

# In[3]:


import numpy as np
from math import*
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import argparse
import multiprocessing as mp
parser = argparse.ArgumentParser(description='Job launcher')
parser.add_argument("-p", type=float) #noise variance
parser.add_argument("-r", type=float) #gamma kernel parameter
parser.add_argument("-k", type=str) #type of kernel
parser.add_argument("-v", type=int) #sample range
parser.add_argument("-d", type=str) #dataset


args = parser.parse_args()

samplerange=np.logspace(1,4,12)
samples=samplerange[args.v]
gamma=args.r #RBF inverse variance
sigma=args.p
kernel=args.k

dataset=args.d

#We download the dataset and convert them in (flattened) numpy arrays, substract the mean column-wise and divide by the standard deviation. The labels y are generated from the 
#task at hand. For example, for even odd MNIST, label y=+1 are assigned to even numbers and y=-1 to odd numbers. These arrays should be stored in a folder datasets/

X=np.load("datasets/{}_X.npy".format(dataset))
y=np.load("datasets/{}_y.npy".format(dataset))

y+=np.random.randn(y.shape[0])*sigma

#creating custom score function for the CV grid search
def custom_score(yhat,y):
    return np.mean((yhat-y)**2)

from sklearn.metrics import make_scorer
quad_error = make_scorer(custom_score, greater_is_better=False)


# In[24]:


def get_errors(samples, X, y, seed):
    global lamb0, Grid

    samples=int(round(samples))

    

    _, p = X.shape
    np.random.seed(seed)
    inds = np.random.choice(range(X.shape[0]), size=samples, replace=True)
    print(inds[:5])
    X_train = X[inds, :] # training data
    y_train = y[inds] # training labels

    X_test = X # test data
    y_test = y # test labels
    
    if kernel=="polynomial":
        KRR= GridSearchCV(KernelRidge(kernel=kernel,degree=5,gamma=gamma),
                param_grid={"alpha": Grid_}
                        ,scoring=quad_error)
    else:
        KRR= GridSearchCV(KernelRidge(kernel=kernel,gamma=gamma),
                    param_grid={"alpha": Grid_}
                            ,scoring=quad_error)
    KRR.fit(X_train,y_train)
    
    yhat_test=KRR.predict(X_test)
    yhat_train=KRR.predict(X_train)
    
    train_error=np.mean((yhat_train-y_train)**2)
    test_error=np.mean((yhat_test-y_test)**2)-sigma**2
    
    best_lamb=KRR.best_params_["alpha"]
    print("Samples{} sigma{} gamma{} test_error{} lamb_opt{}".format(samples,sigma,gamma,test_error,best_lamb))
    return (dataset,kernel,train_error,test_error,best_lamb,seed,samples,samples/p,sigma,gamma)






Grid_=np.array([0]+list(np.logspace(-8,3,500)))

pool = mp.Pool(mp.cpu_count())
results = pool.starmap_async(get_errors, [(samples,X,y,seed) for seed in range(10)]).get()
pool.close()
results=np.array(results)
Df=pd.DataFrame(data=results,columns=["dataset","kernel","train_error","test_error","lambda","seed","samples","alpha","sigma","gamma"])
Df.to_csv("Data/{}/lambda_opt/{}_{}_gamma{}_sigma{}_samples{}.csv".format(dataset,dataset,kernel,gamma,sigma,samples))






