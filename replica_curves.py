# Replica (solid) curves in Fig. 2, 3, 4


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import erf

from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split

from sklearn.kernel_ridge import KernelRidge
import sys


#The following packages can be downloaded from https://github.com/IdePHICS/GCMProject/tree/main/state_evolution and put in the folder g3m_utils/
sys.path.insert(1, 'g3m_utils/')
from state_evolution.data_models.custom import CustomSpectra
from state_evolution.experiments.learning_curve import CustomExperiment
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model




# Dimensions
p = 100000
k = p

 #Noise variance
gamma = k/p

# Regularisation
lamb = 0


replicas={"lamb":[],"samples":[],"test_error":[], "a":[],"b":[],"c":[],"lamb0":[],"sigma":[],"p":[]}



b = 1.5 #called \alpha in the paper
a = 3  # r=(a-1)/2 in the paper
c=1  #called \ell-1 in the paper
lamb0=1#10**(-4)
sigma=1e-3
spec_Omega = np.array([p/(k+1)**b for k in range(p)])
Phi = spec_Omega
Psi = spec_Omega

teacher = np.sqrt(np.array([p/(k+1)**a for k in range(p)]) / spec_Omega)


rho = np.mean(Psi * teacher**2)
diagUtPhiPhitU = Phi**2 * teacher**2

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


Df=pd.DataFrame(replicas)


