
#%%
#Import relevant packages 
from re import S
from turtle import update
from sklearn.cluster import k_means

#For Variational Inference 
import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn
from tqdm import trange

#My packages
from sem import *
from mccode import * 

#Tensorboard 
from torch.utils.tensorboard import SummaryWriter

#For Visualisation and Sampling 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

#for mcmc
import arviz as az

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
lam = torch.tensor([0.8, 0.5])
psi_sqrt = torch.tensor([3.1, 2.2, 1.1])
psi = torch.square(psi_sqrt)

#Generate Latent Variables
eta = torch.randn(N)*sig
lam1 = torch.tensor([1.0])
lam_full= torch.cat((lam1, lam))

#Set Optim Params
iter = 100000
iters = trange(iter, mininterval = 1)
lr = 0.01 

#Set MC Params
num_chains = 4
num_warmup = 1000
num_samples = 10000

lr_nl= 0.01
lr_ps= 0.01
lr_eta = 0.01
#psi and sigma are very slow to converge 

writer = SummaryWriter("test")

#Fix degenerates
degenerate = {} #degenerate lam is 2 dimensional

#Concatenate
hyper = {"sig2_shape": sig2_shape, "sig2_rate": sig2_rate, "psi_shape": psi_shape, "psi_rate": psi_rate, "nu_sig2": nu_sig2, "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}

# %%
#Generate y values based on User Input
# yi ~ id Normal(nu + eta_i * lam, diag(psi)), yi /in R^m
#cov:
like_dist_cov = torch.diag(psi) #m*m tensor 
#means: want a n*m vector of means
like_dist_means = torch.matmul(eta.unsqueeze(1), lam_full.unsqueeze(0)) + nu

#Generate yi
y_data = mvn(like_dist_means, covariance_matrix= like_dist_cov).rsample() #n*m tensor

# %%
#Instantiate SEM model
sem_model = sem_model(y_data = y_data, \
    degenerate= degenerate, hyper= hyper)
# %%
#Instantiate Optimizer Object with uniform learning rate 
# optimizer = torch.optim.Adam([sem_model.qvar[key].var_params for key in sem_model.qvar], lr = lr)

#Instantiate Optimizer Object with different learning rates for parameter groups
optimizer = torch.optim.Adam([{'params': [sem_model.qvar['nu'].var_params, sem_model.qvar['lam'].var_params], 'lr': lr_nl},\
     {'params': [sem_model.qvar['psi'].var_params, sem_model.qvar['sig2'].var_params], 'lr': lr_ps},\
         {'params':[sem_model.qvar['eta'].var_params], 'lr': lr_eta} 
         ])
# %%
#Main Part 1: ADVI
for t in iters:
    # print("psi_params", sem_model.qvar['psi'].var_params)

    optimizer.zero_grad()
    loss = -sem_model.elbo()
    loss.backward()
    optimizer.step()

    # print("psi_grad", sem_model.qvar['psi'].var_params.grad)

    writer.add_scalar(tag = "training_loss", scalar_value=\
                      loss.item(), global_step = t)

    writer.add_scalars("nu and lam",{\
                    'nu1_sig': sem_model.qvar['nu'].var_params[1][0].exp().item(),\
                    'nu2_sig': sem_model.qvar['nu'].var_params[1][1].exp().item(), \
                    'nu3_sig': sem_model.qvar['nu'].var_params[1][2].exp().item(),\
                    'nu1_mean': sem_model.qvar['nu'].var_params[0][0].item(),\
                    'nu2_mean': sem_model.qvar['nu'].var_params[0][1].item(), \
                    'nu3_mean ': sem_model.qvar['nu'].var_params[0][2].item(), \
                    'lambda2_mean': sem_model.qvar['lam'].var_params[0][0].item(),\
                    'lambda2_sig': sem_model.qvar['lam'].var_params[1][0].exp().item(),\
                    'lambda3_sig': sem_model.qvar['lam'].var_params[1][1].exp().item(),\
                    'lambda3_mean': sem_model.qvar['lam'].var_params[0][1].item()}, global_step = t)
    
    writer.add_scalars("psi and sig2", {\
                    'psi_1_alpha': sem_model.qvar['psi'].var_params[0][0].exp().item(),\
                    'psi_2_alpha': sem_model.qvar['psi'].var_params[0][1].exp().item(),\
                    'psi_3_alpha': sem_model.qvar['psi'].var_params[0][2].exp().item(),\
                    'psi_1_beta': sem_model.qvar['psi'].var_params[1][0].exp().item(), \
                    'psi_2_beta': sem_model.qvar['psi'].var_params[1][1].exp().item(), \
                    'psi_3_beta': sem_model.qvar['psi'].var_params[1][2].exp().item(), \
                    'sig2_alpha': sem_model.qvar['sig2'].var_params[0].exp().item(),\
                    'sig2_beta': sem_model.qvar['sig2'].var_params[1].exp().item(),\
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
# %%
# %%
#Prepare data for MCMC
data = {"y": y_data.clone().numpy(),\
        "N": y_data.size(0),\
        "M": y_data.size(1)}
h = {var:param.item() for var,param in hyper.items()}
data.update(h)
# %%
#Main Part 2: Do MCMC
posterior = mc(data)
fit = posterior.sample(num_chains = 4, num_warmup = 1000, num_samples = 10000)
fitp = fit.to_frame() #convert to pandas data frame

# %%
#Comparative Visualisation: Non -Eta Variables 
#Make var string ##optimise later 
var = ['nu.1', 'nu.2', 'nu.3', 'lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']  #hard coded order: nu, lam, psi,sig2

#Sample MC data excluding eta--> pd.df
mcdf = fitp[var]

#Sample VB data excluding eta ---< pd.df
num_sample = torch.tensor([10000])
vb_sample = np.concatenate([sem_model.qvar[key].dist().rsample(num_sample).detach().numpy() for key in sem_model.qvar if key!= 'eta'], axis = 1)
vbdf = pd.DataFrame(vb_sample, columns = var)

#Plot Excluding Eta 
fig, ax = plt.subplots(5,2, constrained_layout = True, figsize = (10,10)) 
fig.delaxes(ax[4,1])
fig.suptitle("Estimated Posterior Densities for Non-Latent,  N =" + str(N))

#manually add in legend
green_patch = mpatches.Patch(color='green', label='MCMC app post.')
blue_patch = mpatches.Patch(color='blue', label='ADVI app post.')
fig.legend(handles=[green_patch, blue_patch], loc = 'lower left')

for v,a in zip(var,ax.flatten()):
    sns.histplot(data = mcdf[v], ax = a, color = 'green', stat = 'density', kde = True)
    sns.histplot(data = vbdf[v], ax = a, stat = 'density', color = 'blue', bins = 100, kde = True)
fig.save_fig("Test Run with Simulated Data, N=" + str(N) + "vb_numiter=" + str(iter), "mcnum_iter= " + str())
