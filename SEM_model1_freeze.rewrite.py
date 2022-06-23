#%%
#Import relevant packages 
from re import S
from turtle import update
from sklearn.cluster import k_means
import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
import math
import pandas as pd
import numpy as np

# %% 
#User to change: 
# Set Hyperparameters, True Param Values, Optimization Parameters

#Set Hyper-parameters 
#sig_2 ~ InvGamma
sig2_shape = torch.tensor([1.0])  
sig2_rate = torch.tensor([1.0])  

#psi ~ iid Inv Gamma for j = 1..m 
psi_shape = torch.tensor([1.0])  
psi_rate = torch.tensor([1.0])  

#nu ~ iid Normal for j = 1...m
nu_sig2 = torch.tensor([10.0])  
nu_mean = torch.tensor([0.0])

#lam_j | psi_j ~ id Normal(mu, sig2*psi_j)
lam_mean = torch.tensor([1.0])
lam_sig2 = torch.tensor([10.0])

#Set True Values for Parameters 
N = 1000 
M = 3
nu = torch.tensor([5.0, 10.0, 2.0])
sig2 = torch.tensor([1.2 * 1.2])
lam = torch.tensor([0.8, 0.5])
psi_sqrt = torch.tensor([3.1, 2.2, 1.1])
psi = torch.square(psi_sqrt)

degenerate = {'psi': psi, 'sig2': sig2}

#Set Optim Params
lr, max_iter = 0.01, 10000
writer = SummaryWriter()

# %% 
# Generate True Values for Parameters based on User Input

#Generate Latent Variables
eta = torch.rand(N)*torch.sqrt(sig2)
lam_1_fixed = torch.tensor([1.0])
lam = torch.cat((lam_1_fixed, lam))

#Concatenate
hyper = {"sig2_shape": sig2_shape, "sig2_rate": sig2_rate, "psi_shape": psi_shape, "psi_rate": psi_rate, "nu_sig2": nu_sig2, "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}

# %%
#Generate y values based on User Input
# yi ~ id Normal(nu + eta_i * lam, diag(psi)), yi /in R^m
#cov:
like_dist_cov = torch.diag(psi) #m*m tensor 
#means: want a n*m vector of means
like_dist_means = torch.matmul(eta.unsqueeze(1), lam.unsqueeze(0)) #n*m tensor
#Generate yi
y_data = mvn(like_dist_means, covariance_matrix= like_dist_cov).rsample() #n*m tensor

# %%
#Create qvar objects.
class qvar_normal():
    def __init__(self, size, mu=None, log_s=None):
        if mu is None:
            mu = torch.randn(size)
        if log_s is None:
            log_s = torch.randn(size)  # log of the standard deviation
        # Variational parameters
        self.var_params = torch.stack([mu, log_s])
        self.var_params.requires_grad = True
        self.size = size
    def dist(self):
        return torch.distributions.Normal(self.phi[0], self.phi[1].exp())
    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)
    def log_prob(self, x):
        return self.dist().log_prob(x).sum()

class qvar_degenerate():
    def __init__(self, values):
        self.var_params = values
        self.var_params.requires_grad = False #do not update this parameter
    def dist(self):
        return "Degenerate, see values attribute"
    def rsample(self):
        return self.var_params.clone()
    def log_prob(self, x):
        return torch.tensor([0.0])

class qvar_invgamma(): #dummy class
    def __init__(self):
        self.var_params = None
        return #do nothing
    def log_prob(self, x):
        return torch.tensor([0.0])

# %%
#Create InvGamma Distribution 
class InvGamma(): #dummy class 
    def __init__(self):
        return #do nothing
    def log_prob(self, x):
        return torch.tensor([0.0])

#%% [markdown]
#Next, create the 'sem_model' class.
#Rationale is to have all information about the model contained in this class. 

#Inputs, as torch tensors: 

# y_data (n*m) dataset
#degenerates: strings from lam, nu,psi, eta, sig2: key: float tensor values 
#hyper: dictionary with string keys for prior distribution parameters

#attributes: 

#y_data, n, m, degenerates, hyper, var_parameters

#functions

#generate_theta_sample(self):
#out: dictionary of var_parameter: torch.tensor with correct dimensions, either [n] or [m]
#log_like(self, theta_sample):
#out: log_likelihood (torch.tensor)

#log_prior(self, theta_sample):
#out: log_prior(torch.tensor)

#neg_entropy(self, theta_sample)
# %%
class sem_model():
    def __init__(self, y_data, degenerate, hyper):
        #user input:
        self.y_data = y_data
        self.degenerate = {var: qvar_degenerate(value) for (var,value) in degenerate.items()}
        self.hyper = hyper
        self.n = y_data.size(0)
        self.m = y_data.size(1)
        self.lam1 = torch.tensor([1.0])
        
        #hardcoded distributional information
        #qvar_family
        self.qvar = {'nu': qvar_normal(self.m), 'lam': qvar_normal((self.m)-1),
        'eta': qvar_normal(self.n), 'psi': 
        qvar_invgamma(), 'sig2': qvar_invgamma()}
        #update 
        self.qvar.update(self.degenerate)

    def generate_theta_sample(self):
        theta_sample = {var: qvar.rsample() for (var,qvar) in self.qvar.items()}
        return theta_sample

    def log_like(self,theta_sample):
        like_dist_cov = torch.diag(theta_sample['psi'])
        lam_full = torch.cat((self.lam1, theta_sample['lam']))
        like_dist_means = torch.matmul(theta_sample['eta'].unsqueeze(1), lam_full.unsqueeze(0))
        log_like = mvn(like_dist_means, covariance =like_dist_cov).log_prob(self.y_data).sum()

        return log_like

    def log_prior(self, theta_sample): 
        #hard coded prior 
        priors = {'nu': Normal(loc = self.hyper['nu_mean'], scale = torch.sqrt(self.hyper['nu_sig2'])), \
        'sig2': InvGamma(),\
        'psi': InvGamma(),\
        'eta': Normal(0, torch.sqrt(theta_sample['sig2'])),\
        'lam': Normal(loc = self.hyper['lam_mean'], \
            scale = torch.sqrt(self.hyper['lam_sig2']*(theta_sample['psi'][1:])))
            }

        log_priors = {var: priors[var].log_prob(theta_sample[var]).sum() for var in priors if var not in self.degenerate}

        return sum(log_priors.values())

    def entropy(self, theta_sample):
        qvar_prob = {var: self.qvar[var].log_prob(sample) for (var,sample) in theta_sample.items()}

        return sum(qvar_prob.values())
# %%
#elbo function 

def elbo(sem_model):
    theta_sample = sem_model.generate_theta_sample()

    return sem_model.log_like(theta_sample) + sem_model.log_prior(theta_sample) - sem_model.entropy(theta_sample)

# %%
#Instantiate SEM model
sem_model = sem_model(y_data = y_data, \
    degenerate= degenerate, hyper= hyper)
# %%
#Instantiate Optimizer Object
optimizer = torch.optim.Adam([sem_model.qvar[key].var_params for key in sem_model.qvar], lr = lr)
iters = trange(max_iter, mininterval = 1)

# %%
#What results do we want to visualise?

#only create for non-degenerate distributions 
#1. #{'nu': [nu_mean, nu_sig], 'lam': [lam_mean, lam_sig], 'eta': ['eta_mean, eta_sigma']}
#2. expand list items out into scalar dicts  
#3. plot/visualise in an appropriate way 
#4. How to add maximum functionality?

# %%
#Hardcode values to record for every parameter. This is clean but really annoying to have to do for every single one  
#nu 
nu_mean = {'nu_'+ str(j+1) + 'mean': sem_model.qvar['nu'].var_params[0][j].item() for j in range(sem_model.qvar['nu'].size)}
nu_sig = {'nu_'+ str(j+1) + 'sig': sem_model.qvar['nu'].var_params[1][j].exp().item() for j in range(sem_model.qvar['nu'].size)}

#lam 
lam_mean = {'lam_'+ str(j+2) + 'mean': sem_model.qvar['lam'].var_params[0][j].item() for j in range(sem_model.qvar['lam'].size)}

lam_sig = {'lam_'+ str(j+2) + 'sig': sem_model.qvar['lam'].var_params[1][j].exp().item() for j in range(sem_model.qvar['lam'].size)}

#eta 
eta_mean = {'eta_'+ str(j+1) + 'mean': sem_model.qvar['eta'].var_params[0][j].item() for j in range(sem_model.qvar['eta'].size)}

eta_sig = {'eta_'+ str(j+1) + 'sig': sem_model.qvar['eta'].var_params[1][j].exp().item() for j in range(sem_model.qvar['eta'].size)}
# %%
#This aggregation of information is a gargantuan task in itself.
#Spend half a day finding better ways to visualise information using tensorboard.