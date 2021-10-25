#!/usr/bin/env python
# coding: utf-8

# # Ridge regression on the G$^3$M model

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import erf

from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

from sklearn.kernel_ridge import KernelRidge
import sys

sys.path.insert(1, 'g3m_utils/')
from state_evolution.data_models.custom import CustomSpectra
from state_evolution.experiments.learning_curve import CustomExperiment
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model

# %load_ext autoreload
# %autoreload 2


# ## Global Variables

# In[66]:


# Dimensions
p = 100000
k = p

 #Noise variance
gamma = k/p

# Regularisation
lamb = 0


# ## Replicas

# In[67]:


replicas={"lamb":[],"samples":[],"test_error":[], "a":[],"b":[],"c":[],"lamb0":[],"sigma":[],"p":[]}


# In[68]:


b = 1.5
a = 3
c=1
lamb0=1#10**(-4)
sigma=1e-3
spec_Omega = np.array([p/(k+1)**b for k in range(p)])
Phi = spec_Omega
Psi = spec_Omega

teacher = np.sqrt(np.array([p/(k+1)**a for k in range(p)]) / spec_Omega)


# In[69]:


rho = np.mean(Psi * teacher**2)
diagUtPhiPhitU = Phi**2 * teacher**2


# In[70]:


print('Loading data model')
data_model = CustomSpectra(gamma = gamma,
                           rho = rho+sigma**2, 
                           spec_Omega = spec_Omega, 
                           spec_UPhiPhitUT = diagUtPhiPhitU) #rho<- rho+sigma**2 is a conveninent artificial way to include noise

print('Loading experiment')
experiment = CustomExperiment(task = 'ridge_regression', 
                              regularisation = lamb, 
                              data_model = data_model, 
                              tolerance = 1e-16, 
                              damping = 0.5, 
                              verbose = False, 
                              max_steps = 1000)


# In[71]:


def error_lam(log_lam,alpha):
    lam=np.exp(-np.log(10)*log_lam)
    experiment = CustomExperiment(task = 'ridge_regression', 
                              regularisation = lam, 
                              data_model = data_model, 
                              tolerance = 1e-16, 
                              damping = 0.3, 
                              verbose = False, 
                              max_steps = 1000)
    experiment.learning_curve(alphas = np.array([alpha]))
    error=experiment.get_curve()["test_error"][0]
    return error
    


# In[72]:


alphas = np.logspace(.8, 5, 15)
from scipy.optimize import minimize_scalar



CV=True

for alpha in alphas:
    print("samples",alpha)
    if CV:
        minimization=minimize_scalar(error_lam, (-10,10),args=(alpha/p),tol=1e-16)
        e_g,log_lamb=minimization.fun,minimization.x
    else:
        log_lamb=(np.log(alpha)*c)/np.log(10)-np.log10(lamb0)
        e_g=error_lam(log_lamb,alpha/p)
    
    print(e_g-sigma**2,log_lamb)
    replicas["lamb"].append(10**(-log_lamb))
    replicas["samples"].append(alpha)
    replicas["test_error"].append(e_g-sigma**2)#generalisation gap
    
    replicas["a"].append(a)
    replicas["b"].append(b)
    replicas["lamb0"].append(lamb0)
    replicas["c"].append(c)
    replicas["p"].append(p)
    replicas["sigma"].append(sigma)


# In[73]:


Df=pd.DataFrame(replicas)
Df=Df[Df["test_error"]>0]


# ### Plotting

# In[159]:



colors=["blue","red","green","orange","orchid","cornflowerblue","black"]
color_index=0
for sigma in Df.sigma.unique():
    #if sigma==10**(0.4):continue
    rep=Df[Df["sigma"]==sigma][(Df["samples"]>10**.8)&(Df["samples"]<=10**(6.5))]
    col=colors[color_index]
    plt.loglog(rep["samples"],rep["test_error"],c=col,label=r"$\sigma={}$".format(round(sigma,4)))
    #plt.loglog(rep["samples"],rep["test_error_0"],ls="",marker=".",c=col)
    
    
    #plt.ylim(10**(-14),1)
    
    color_index+=1
    
    

    if CV:
        expo1=-min(a-1,2*b)
        expo2=-1+1/min(a,2*b+1)
        limit=np.log(sigma**(-2/min(a-1,2*b)))/np.log(10)
    else:
        if c==np.inf:
            expo1=-min(a-1,2*b)
            expo2=0
            limit=np.log(sigma**(-2/min(a-1,2*b)))/np.log(10)
        else:
            expo1=-min(a-1,2*b)*(1+c)/b
            expo2=-(b-c-1)/b
            limit=np.log(sigma**(2/(1-(1+c)/b*min(a,2*b+1))))/np.log(10)
            print(limit)
    
    x1=10**(np.linspace(.8,max(.8,limit),10))
    y1=10**(np.log(x1/x1[0])/np.log(10)*(expo1))*np.array(rep["test_error"])[0]/1.2
    x2=10**(np.linspace(max(.8,limit),5,10))
    y2=10**(np.log(x2/x2[-1])/np.log(10)*(expo2))*np.array(rep["test_error"])[-1]/1.1

    plt.loglog(x1,y1,ls="--",c=col,alpha=0.5)#,label=r'$10^{-\mathrm{min}(a-1,2b)}$')
    plt.loglog(x2,y2,ls="--",c=col,alpha=0.5)#,label=r'$10^{\frac{1}{\mathrm{min}(a,2b+1)}-1}$')

    """
    plt.annotate(r'$-\mathrm{min}(a-1,2b)$', # this is the text
                             (x1[round((len(x1))/3)],y1[round((len(x1))/2)]), # this is the point to label
                             textcoords="offset points", # how to position the text
                             xytext=(0,-30), # distance from text to points (x,y)
                             ha='center', # horizontal alignment can be left, right or center
                             c="g") 
    plt.annotate(r'$\frac{1}{\mathrm{min}(a,2b+1)}-1$', # this is the text
                             (x2[round((len(x2))/3)],y2[round((len(x2))/2)]), # this is the point to label
                             textcoords="offset points", # how to position the text
                             xytext=(0,-20), # distance from text to points (x,y)
                             ha='center', # horizontal alignment can be left, right or center
                             c="r") 

    """
plt.legend(loc=3)
plt.xlabel(r"$n$")
plt.xlim(10,10**(5))
plt.ylim(10**(-18),1)
if CV:
    plt.ylabel(r"$\epsilon_g^\star-\sigma^2$")
    plt.title(r" $a={}~~b={}$".format(a,b))
else:
    if c==np.inf:
        plt.ylabel(r"$\epsilon_g-\sigma^2$")
        plt.title(r" $a={}~~b={}~~\lambda=0$".format(a,b,0))
    else:
        plt.ylabel(r"$\epsilon_g-\sigma^2$")
        plt.title(r" $a={}~~b={}~~c={}$".format(a,b,c))


# In[126]:


10**(4.5)


# In[392]:


colors=["blue","red","green","orange","orchid","cornflowerblue","black"]
color_index=0
for sigma in Df.sigma.unique():
    
    col=colors[color_index]
    rep=Df[Df["sigma"]==sigma]
    plt.loglog(rep["samples"],rep["lamb"],c=col,label=r"$\sigma={}$".format(round(sigma,4)))
    
    limit=np.log(sigma**(-2/min(a-1,2*b)))/np.log(10)

    x1=10**(np.linspace(limit,5,10))
    y1=10**(np.log(x1/x1[-1])/np.log(10)*(-b/min(a,2*b+1)))*np.array(rep["lamb"])[-1]/1.6
    
    plt.loglog(x1,y1,ls="--",c=col,alpha=0.5)
    color_index+=1
plt.ylim(1e-10,10000)
plt.xlim(5,10**5)
plt.xlabel(r"$n$")
plt.ylabel(r"$\lambda^\star$")
plt.title(r" $a={}~~b={}$".format(a,b))
plt.legend()


# In[433]:


if CV:
    Df.to_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}.csv".format(a,b), index=False)
else:
    if c==np.inf:
        Df.to_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}_noreg.csv".format(a,b), index=False)
    else:
        Df.to_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}_c{}.csv".format(a,b,c), index=False)


# ## Simulations

# Ran on cluster, see .py file in ./

# In[73]:


CV=True
a=1.5
b=1.2


# In[74]:


if CV:
    rep=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}.csv".format(a,b))
else:
    rep=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}_noreg.csv".format(a,b))


# In[75]:


sigma=rep["sigma"].unique()[0]


# In[77]:


def get_samples(*, samples, teacher,sig):    
    X = np.sqrt(spec_Omega) * np.random.normal(0,1,(samples, p))
    y = X @ teacher / np.sqrt(p)+sig*np.random.normal(0,1,(samples,))
    
    return X/np.sqrt(p), y


# In[78]:


use_replica_lambda=False


def ridge_estimator(X, y, lamb):
    '''
    Implements the pseudo-inverse ridge estimator.
    '''
    m, n = X.shape
    if m >= n:
        return np.linalg.inv(X.T @ X + lamb*np.identity(n)) @ X.T @ y
    elif m < n:
        return X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(m)) @ y

def get_errors(seeds=10, * , samples, lamb, teacher):
    '''
    Get averaged training and test error over a number of seeds for a fixed number of samples
    Input:
        - samples: # of samples
        - gram: gram matrix of the total data set (train+test)
        - y: labels of the total data set
        - seeds: number of seeds
        - trick: if true, evaluate test set over whole data set (test and train)
    Return:
        - 4-uple with (avg train error, std train error, avg test error, std test error)
    '''
    
    eg, et = [], []
    for i in range(seeds):
        print('Seed: {}'.format(i))
        X_train, y_train = get_samples(samples = samples,
                                       teacher = teacher,
                                      sig=sigma)
        
        X_test, y_test = get_samples(samples = samples,
                                     teacher = teacher,
                                    sig=sigma)
        if use_replica_lambda==True or CV==False:
            w = ridge_estimator(X_train, y_train, lamb=lamb)
            yhat_train = X_train @ w 
            yhat_test = X_test @ w 
        else:
            
            reg = linear_model.RidgeCV(alphas=np.logspace(np.log10(lamb)-2,np.log10(lamb)+2,20))
            reg.fit(X_train,y_train)
    
            yhat_test=reg.predict(X_test)
            yhat_train=reg.predict(X_train)

        train_error = np.mean((y_train - yhat_train)**2)
        test_error = np.mean((y_test - yhat_test)**2)-sigma**2

        eg.append(test_error)
        et.append(train_error)
        #print(test_error)

    return (np.mean(et), np.std(et), np.mean(eg), np.std(eg))

def simulate(lamb = 0.01, seeds = 10, *, sample_range, teacher):
    
    
    
    for samples in sample_range:
        print(rep[(np.abs(rep["samples"]-samples)<1) & (rep["a"]==a) & (rep["b"]==b)&(rep["sigma"]==sigma)]["lamb"])
        lamb=float(rep[(np.abs(rep["samples"]-samples)<1) & (rep["a"]==a) & (rep["b"]==b)&(rep["sigma"]==sigma)]["lamb"])
        print('Simulating sample: {} lambda {}'.format(samples,lamb))
        et, et_std, eg, eg_std = get_errors(samples=int(samples), 
                                            lamb=lamb, 
                                            seeds=seeds, 
                                            teacher=teacher)
        data['samples'].append(samples)
        data['sample_complexity'].append(samples/p)
        data['task'].append('regression')
        data['loss'].append('l2')

        
        
        data['lambda'].append(lamb)
        data['test_error'].append(eg)
        data['test_error_std'].append(eg_std)
        data['train_error'].append(et)
        data['train_error_std'].append(et_std)
        data["a"].append(a)
        data["b"].append(b)
        print(eg)
        

   


# In[ ]:





# In[79]:



data = {'test_error': [], 'train_error': [], 'test_error_std': [], 
            'train_error_std': [], 'lambda': [],
            'sample_complexity': [], 'samples': [], 'task': [], 'loss': [],"a":[],"b":[]}


# In[80]:


rho = np.mean(Psi * teacher**2)

spec_Omega = np.array([p/(k+1)**b for k in range(p)])
Phi = spec_Omega
Psi = spec_Omega

teacher = np.sqrt(np.array([p/(k+1)**a for k in range(p)]) / spec_Omega)


simulate(lamb = 0.01, seeds = 10, sample_range= np.logspace(.8, 5, 15), teacher=teacher)


# ## Plots

# In[36]:


a=4.0
b=1.5
c=0
CV=True
from os import listdir
import pandas as pd


# In[37]:


if CV:
    path="Data/Simulations/Artificial/CV"
else:
    if c==np.inf:
        path="Data/Simulations/Artificial/no_reg"
    else:
        path="Data/Simulations/Artificial/decay"
results_files=listdir(path)
Data=pd.read_csv(path+"/"+results_files[0], index_col=0)


#Data["sigma"]=read_sigma(results_files[0])
#Data["gamma"]=read_gamma(results_files[0])
for file in results_files[1:]:
    new_line=pd.read_csv(path+"/"+file, index_col=0)
    #print(new_line)
    Data=pd.concat([Data,new_line])


# In[38]:


if CV:
    Data.to_csv("Data/Simulations/Artificial/Aggregated/Artificial_CV.csv",index=False)
else:
    if c==np.inf:
        Data.to_csv("Data/Simulations/Artificial/Aggregated/Artificial_no_reg.csv",index=False)
    else:
        Data.to_csv("Data/Simulations/Artificial/Aggregated/Artificial_decay.csv",index=False)


# In[41]:


a=4.0
b=2.5
c=0.75
CV=False
if CV:
    Data=pd.read_csv("Data/Simulations/Artificial/Aggregated/Artificial_CV.csv")
    replicas=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}.csv".format(a,b))
else:
    if c==np.inf:
        Data=pd.read_csv("Data/Simulations/Artificial/Aggregated/Artificial_no_reg.csv")
        replicas=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}_noreg.csv".format(a,b))
    else:
        Data=pd.read_csv("Data/Simulations/Artificial/Aggregated/Artificial_decay.csv")
        replicas=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}_c{}.csv".format(a,b,c))
        
        
Data=Data[(Data["a"]==a)&(Data["b"]==b)]
Data=Data.sort_values(by=["samples"])
Data=Data.groupby(["samples","sigma"],as_index=False).mean()
colors=["green","blue","red","orchid","cornflowerblue"]
i=0
for sigma in Data["sigma"].unique():
    c=colors[i]
    print(sigma)
    selection=Data[Data["sigma"]==sigma]
    repl=replicas[replicas["sigma"]==sigma]
    plt.loglog(selection["samples"],selection["test_error"],ls="", marker=".",c=c)
    plt.loglog(repl["samples"],repl["test_error"],c=c)
    i+=1

plt.xlim(8,100000)


# ## sklearn CV

# In[24]:


a=2.5
b=2.0
c=0
CV=True
from os import listdir
import pandas as pd


# In[25]:


path="Data/Simulations/Artificial/sklearn"
results_files=listdir(path)
Data=pd.read_csv(path+"/"+results_files[0], index_col=0)


#Data["sigma"]=read_sigma(results_files[0])
#Data["gamma"]=read_gamma(results_files[0])
for file in results_files[1:]:
    new_line=pd.read_csv(path+"/"+file, index_col=0)
    #print(new_line)
    Data=pd.concat([Data,new_line])


# In[26]:


Data.to_csv("Data/Simulations/Artificial/Aggregated/Artificial_sklearnCV.csv",index=False)


# In[27]:


Data=pd.read_csv("Data/Simulations/Artificial/Aggregated/Artificial_sklearnCV.csv")
replicas=pd.read_csv("./Data/Replica/Noisy_opt_lambda_ridge_decay_a{}_b{}.csv".format(a,b))

fig=plt.figure(figsize=(5,8))
ax1=plt.subplot(211)
ax2=plt.subplot(212, sharex = ax1)




#print(Data[(np.abs(Data["sigma"]-1)<0.01)&(np.abs(Data["samples"]-200)<50)])      
        
Data=Data[(Data["a"]==a)&(Data["b"]==b)]
Data=Data.sort_values(by=["samples"])
Data=Data.groupby(["samples","sigma"],as_index=False).mean()
colors=["green","blue","cornflowerblue","red","red","red"]
i=0

for sigma in Data["sigma"].unique():
    c=colors[i]
    #print(sigma)
    selection=Data[Data["sigma"]==sigma]
    repl=replicas[replicas["sigma"]==sigma]
   
    #print(selection)
    
    ax1.loglog(selection["samples"],selection["test_error"],ls="", marker=".",c=c)
    ax1.loglog(repl["samples"],repl["test_error"],c=c,label=r"$\sigma={}$".format(np.round(sigma,4)))
    
    ax2.loglog(repl["samples"],repl["lamb"]/repl["samples"],c=c)
    ax2.loglog(selection["samples"],selection["lambda"]/selection["samples"],marker=".",ls="",c=c)
    i+=1
    

ax1.legend()
ax1.set_xlim(8,20000)
ax1.set_ylim(10**(-10),1)

ax1.set_ylabel(r"$\epsilon_g^\star-\sigma^2$")
ax1.set_title(r"$\alpha={},~ r={},~ \ell^\star$".format(b,round((a-1)/2/b,3),2.5))

ax2.set_xlabel(r"$n$")
ax2.set_ylabel(r"$\lambda^\star$")
ax2.set_ylim(1e-13,1)
ax2.set_xlim(25,10000)

plt.subplots_adjust(hspace=0.0)

plt.savefig('./Plots/Paper_plots/Artificial_a{}_b{}_CV.pdf'.format(a,b), bbox_inches='tight')


# In[ ]:





# In[ ]:




