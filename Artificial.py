#!/usr/bin/env python
# coding: utf-8

# # 3-layer on GAN data

# In[ ]:

from math import*
import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp
from sklearn import linear_model
parser = argparse.ArgumentParser(description='Job launcher')
parser.add_argument("-a", type=float)#a
parser.add_argument("-b", type=float)#b
parser.add_argument("-c",type=float)#c
parser.add_argument("-s", type=float)#logsigma
parser.add_argument("-p", type=int)#feature space dim
parser.add_argument("-v", type=int)#samples
parser.add_argument("-o", type=str)#CV or noreg or decay
parser.add_argument("-l", type=str)#either "use_replica_lambda" or "sklearn"

args = parser.parse_args()


a=args.a
b=args.b
c=args.c
p=int(args.p)

spec_Omega = np.array([p/(k+1)**b for k in range(p)])
Phi = spec_Omega
Psi = spec_Omega

sigma=10**(args.s)


teacher = np.sqrt(np.array([p/(k+1)**a for k in range(p)]) / spec_Omega)
rho = np.mean(Psi * teacher**2)
diagUtPhiPhitU = Phi**2 * teacher**2

if args.o=="CV":
    CV=True
else:
    CV=False



if args.o=="CV":
    rep=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}.csv".format(a,b))
elif args.o=="no_reg":
    rep=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}_noreg.csv".format(a,b))
elif args.o=="decay":
    rep=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}_c{}.csv".format(a,b,c))
rep=rep[(np.abs(rep["sigma"]/sigma-1)<0.01)& (rep["a"]==a) & (rep["b"]==b)]

samples=int(round(np.array(rep["samples"])[int(args.v)]))
use_replica_=args.l
if use_replica_=="use_replica_lambda":
    use_replica_lambda=True
else:
    use_replica_lambda=False

lamb=rep[(np.abs(rep["samples"]-samples)<1)]["lamb"].mean()

def get_samples( samples, teacher,seed):
    np.random.seed(seed)    
    X = np.sqrt(spec_Omega) * np.random.normal(0,1,(samples, p))
    np.random.seed(seed+10)
    y = X @ teacher / np.sqrt(p)+np.random.normal(0,1,(samples,))*sigma
    
    return X/np.sqrt(p), y


def ridge_estimator(X, y, lamb=0.1):
    '''
    Implements the pseudo-inverse ridge estimator.
    '''
    m, n = X.shape
    if m >= n:
        return np.linalg.inv(X.T @ X + lamb*np.identity(n)) @ X.T @ y
    elif m < n:
        return X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(m)) @ y

def get_errors(samples, teacher,seed):
     
    
    X_train, y_train = get_samples(samples,
                                    teacher,seed)
    
    X_test, y_test = get_samples(samples,
                                    teacher,3*seed)

    if use_replica_lambda==True or CV==False:
        w = ridge_estimator(X_train, y_train, lamb=lamb)
        yhat_train = X_train @ w 
        yhat_test = X_test @ w 
        lamb_opt=lamb
    else:
        
        reg = linear_model.RidgeCV(alphas=np.hstack((np.logspace(-5,1.5,100),np.array([0]))))

        reg.fit(X_train,y_train)

        yhat_test=reg.predict(X_test)
        yhat_train=reg.predict(X_train)

        lamb_opt=reg.alpha_

    train_error = np.mean((y_train - yhat_train)**2)
    test_error = np.mean((y_test - yhat_test)**2)-sigma**2 #error gap

 

    return [train_error, test_error, a,b,c,lamb_opt,samples,samples/p,sigma,p,seed]



pool = mp.Pool(mp.cpu_count())
results = pool.starmap_async(get_errors, [(samples,teacher,s) for s in range(0,50)]).get()
pool.close()
results=np.array(results)
Df=pd.DataFrame(data=np.array(results),columns=["train_error","test_error","a","b","c","lambda","samples","sample_complexity","sigma","p","seed"])

if args.o!="decay":
    if use_replica_lambda==False:
        filename="./Data/Artificial/sklearn/Simus_a{}_b{}_samples{}_{}.csv".format(a,b,sigma,samples,args.o)
    else:
        filename="./Data/Artificial/{}/Simus_a{}_b{}_sigma{}_samples{}_{}.csv".format(args.o,a,b,sigma,samples,args.o)
else:
    filename="./Data/Artificial/{}/Simus_a{}_b{}_c{}_sigma{}_samples{}_{}.csv".format(args.o,a,b,c,sigma,samples,args.o)


Df.to_csv(filename)

