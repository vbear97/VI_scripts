#%%
#Import relevant packages 
from re import S
from turtle import update
from sklearn.cluster import k_means

import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import trange
# import math
# import pandas as pd
# import numpy as np

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
sig = torch.tensor([1.2])
sig2= torch.square(sig)
lam = torch.tensor([10.0, 15.0])
psi_sqrt = torch.tensor([3.1, 2.2, 1.1])
psi = torch.square(psi_sqrt)

#Generate Latent Variables
eta = torch.randn(N)*sig

lam_1_fixed = torch.tensor([1.0])
lam = torch.cat((lam_1_fixed, lam))

#Set Optim Params
lr, max_iter = 0.01, 20000
writer = SummaryWriter("test")

#Fix degenerates
degenerate = {'psi': psi, 'sig2': sig2}

#Concatenate
hyper = {"sig2_shape": sig2_shape, "sig2_rate": sig2_rate, "psi_shape": psi_shape, "psi_rate": psi_rate, "nu_sig2": nu_sig2, "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}

# %%
#Generate y values based on User Input
# yi ~ id Normal(nu + eta_i * lam, diag(psi)), yi /in R^m
#cov:
like_dist_cov = torch.diag(psi) #m*m tensor 
#means: want a n*m vector of means
like_dist_means = torch.matmul(eta.unsqueeze(1), lam.unsqueeze(0)) + nu

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
        self.var_params = torch.stack([mu, log_s]) #are always unconstrained
        self.var_params.requires_grad = True
    def dist(self):
        return torch.distributions.Normal(self.var_params[0], self.var_params[1].exp())
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
        return {var: qvar.rsample() for (var,qvar) in self.qvar.items()}

    def log_like(self,theta_sample):
        like_dist_cov = torch.diag(theta_sample['psi'])
        lam_full = torch.cat((self.lam1, theta_sample['lam']))
        like_dist_means = torch.matmul(theta_sample['eta'].unsqueeze(1), lam_full.unsqueeze(0)) + theta_sample['nu']
        log_like = mvn(like_dist_means, covariance_matrix= like_dist_cov).log_prob(self.y_data).sum()
        return log_like

    def log_prior(self, theta_sample): 
        #hard coded prior 
        priors = {'nu': Normal(loc = self.hyper['nu_mean'], scale = torch.sqrt(self.hyper['nu_sig2'])), \
        'sig2': InvGamma(),\
        'psi': InvGamma(),\
        'eta': Normal(loc = 0, scale = torch.sqrt(theta_sample['sig2'])), 
        'lam': Normal(loc = self.hyper['lam_mean'], \
            scale = torch.sqrt(self.hyper['lam_sig2']*(theta_sample['psi'][1:])))
            }

        log_priors = {var: priors[var].log_prob(theta_sample[var]).sum() for var in priors if var not in self.degenerate}

        return sum(log_priors.values())

    def entropy(self, theta_sample):
        qvar_prob = {var: self.qvar[var].log_prob(sample) for (var,sample) in theta_sample.items()}

        return sum(qvar_prob.values())
    
    def elbo(self):
        theta_sample = self.generate_theta_sample()

        return self.log_like(theta_sample) + self.log_prior(theta_sample) - self.entropy(theta_sample)

# %%
#Instantiate SEM model
sem_model = sem_model(y_data = y_data, \
    degenerate= degenerate, hyper= hyper)
# %%
#Instantiate Optimizer Object
optimizer = torch.optim.Adam([sem_model.qvar[key].var_params for key in sem_model.qvar], lr = lr)

# %%
#Optimise
for t in range(max_iter):
    optimizer.zero_grad()
    loss = -sem_model.elbo()
    loss.backward()
    optimizer.step()
    writer.add_scalar(tag = "training_loss: step_size="+str(lr), scalar_value=\
                      loss.item(), global_step = t)

    writer.add_scalars("vp",{\
                       'nu1_sig': sem_model.qvar['nu'].var_params[1][0].exp().item(),\
                        'nu2_sig': sem_model.qvar['nu'].var_params[1][1].exp().item(), \
                        'nu3_sig': sem_model.qvar['nu'].var_params[1][2].exp().item(),\
                        'nu1_mean': sem_model.qvar['nu'].var_params[0][0].item(),\
                        'nu2_mean': sem_model.qvar['nu'].var_params[0][1].item(), \
                        'nu3_mean ': sem_model.qvar['nu'].var_params[0][2].item(), \
                    'lambda2_mean': sem_model.qvar['lam'].var_params[0][0].item(),\
                    'lambda2_sig': sem_model.qvar['lam'].var_params[1][0].exp().item(),\
                    'lambda3_sig': sem_model.qvar['lam'].var_params[1][1].exp().item(),\
                    'lambda3_mean': sem_model.qvar['lam'].var_params[0][1].item(),\
                    'psi_1': sem_model.qvar['psi'].var_params[0].item(),\
                    'psi_2': sem_model.qvar['psi'].var_params[1].item(), \
                    'psi_3': sem_model.qvar['psi'].var_params[2].item(),\
                'sig2': sem_model.qvar['sig2'].var_params[0].item(),\
                    }, global_step = t)
    writer.add_scalars("eta", \
                       {'eta1_mean': sem_model.qvar['eta'].var_params[0][0].item(),\
                        'eta1_sig': sem_model.qvar['eta'].var_params[1][0].exp().item(),\
                        'eta1_true': eta[0].item(),\
                        'eta50_mean': sem_model.qvar['eta'].var_params[0][50].item(),\
                        'eta50_sig': sem_model.qvar['eta'].var_params[1][50].exp().item(),\
                        'eta50_true': eta[50].item(),\
                        'eta100_mean': sem_model.qvar['eta'].var_params[0][99].item(),\
                        'eta100_sig': sem_model.qvar['eta'].var_params[1][99].exp().item(),\
                        'eta100_true': eta[99].item(),\
                        'eta200_mean': sem_model.qvar['eta'].var_params[0][200].item(),\
                        'eta200_sig': sem_model.qvar['eta'].var_params[1][200].exp().item(),\
                        'eta200_true': eta[200].item(),\
                        'eta300_mean': sem_model.qvar['eta'].var_params[0][300].item(),\
                        'eta300_sig': sem_model.qvar['eta'].var_params[1][300].exp().item(),\
                        'eta300_true': eta[300].item(),\
                        'eta400_mean': sem_model.qvar['eta'].var_params[0][400].item(),\
                        'eta400_sig': sem_model.qvar['eta'].var_params[1][400].exp().item(),\
                        'eta400_true': eta[400].item(),\
                        'eta500_mean': sem_model.qvar['eta'].var_params[0][500].item(),\
                        'eta500_sig': sem_model.qvar['eta'].var_params[1][500].exp().item(),\
                        'eta500_true': eta[500].item(),\
                        'eta600_mean': sem_model.qvar['eta'].var_params[0][600].item(),\
                        'eta600_sig': sem_model.qvar['eta'].var_params[1][600].exp().item(),\
                        'eta600_true': eta[600].item(),\
                        'eta750_mean': sem_model.qvar['eta'].var_params[0][750].item(),\
                        'eta750_sig': sem_model.qvar['eta'].var_params[1][750].exp().item(),\
                        'eta750_true': eta[750].item(),\
                        }, global_step = t)
