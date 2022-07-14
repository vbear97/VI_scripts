
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
from mle import *
from mfvb import *
from vis_mcvb import *

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

#for data storage
import pickle

# %%
#User to change: Hyperparameters, Optimization Parameters
#Set Hyper-parameters 
#sig_2 ~ InvGamma
sig2_shape = torch.tensor([0.5])  
sig2_rate = torch.tensor([0.5])  

#psi ~ iid Inv Gamma for j = 1..m 
psi_shape = torch.tensor([0.5])  
psi_rate = torch.tensor([0.005])  

#nu ~ iid Normal for j = 1...m
nu_sig2 = torch.tensor([100.0])  
nu_mean = torch.tensor([0.0])

#lam_j | psi_j ~ id Normal(mu, sig2*psi_j)
lam_mean = torch.tensor([0.0])
lam_sig2 = torch.tensor([1.0])

#Fix degenerates
degenerate = {}
#Concatenate
hyper = {"sig2_shape": sig2_shape, "sig2_rate": sig2_rate, "psi_shape": psi_shape, "psi_rate": psi_rate, "nu_sig2": nu_sig2, "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}

#Set Optim Params
iter = 100000
iters = trange(iter, mininterval = 1)
lr = 0.01 
lr_nl= 0.001
lr_ps= 0.01
lr_eta = 0.01
#psi and sigma are very slow to converge 

#Set MC Params
num_chains = 4
num_warmup = 7500
num_samples = 15000

writer = SummaryWriter("test")
# %%
#Import Holzinger and Swineford data

from semopy.examples import holzinger39
hdata = holzinger39.get_data()
#want only visual p,cubes, lozenges test, in that order
#hdata is a pandas dataframe
myhdata = hdata[['x1', 'x2','x3']]
y_data = torch.tensor(myhdata.values, requires_grad=False)
#but how to verify this is actually the desired data? I don't see any documentation for this. 
#now convert to torch tensor

# %% 
#Simulate y_data

#Set True Values for Parameters 
N = 301
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

#Fix degenerates
degenerate = {} #degenerate lam is 2 dimensional

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
    #need to be able to initialise hyper-parameters of SEM model

# %%
#Instantiate Optimizer Object with uniform learning rate 
# optimizer = torch.optim.Adam([sem_model.qvar[key].var_params for key in sem_model.qvar], lr = lr)

#Instantiate Optimizer Object with different learning rates for parameter groups
optimizer = torch.optim.Adam([{'params': [sem_model.qvar['nu'].var_params, sem_model.qvar['lam'].var_params], 'lr': lr_nl},\
     {'params': [sem_model.qvar['psi'].var_params, sem_model.qvar['sig2'].var_params], 'lr': lr_ps},\
         {'params':[sem_model.qvar['eta'].var_params], 'lr': lr_eta} 
         ])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(\
    optimizer = optimizer,\
    mode = 'min',\
    factor = 0.1,\
    patience = 100,\
)
# %%
#Main Part 1: ADVI
for t in iters:
    # print("psi_params", sem_model.qvar['psi'].var_params)

    optimizer.zero_grad()
    loss = -sem_model.elbo()
    loss.backward()
    optimizer.step()

    scheduler.step()

    # print("psi_grad", sem_model.qvar['psi'].var_params.grad)

    writer.add_scalar(tag = "training_loss", scalar_value=\
                      loss.item(), global_step = t)

    #clean up: nu and lam tensorboard writing -  concise way please
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

#Sample VB data excluding eta ---< pd.df
var = ['nu.1', 'nu.2', 'nu.3', 'lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']
num_sample = torch.tensor([num_chains * num_samples])
vb_sample = np.concatenate([sem_model.qvar[key].dist().rsample(num_sample).detach().numpy() for key in sem_model.qvar if key!= 'eta'], axis = 1)
vbdf = pd.DataFrame(vb_sample, columns = var)

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
fit = posterior.sample(num_chains = 4, num_warmup = num_warmup, num_samples = num_samples, delta = 0.85)
fitp = fit.to_frame() #convert to pandas data frame
var = ['nu.1', 'nu.2', 'nu.3', 'lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']  #hard coded order, not efficient: nu, lam, psi,sig2

#Sample MC data excluding eta--> pd.df
mcdf = fitp[var]
# %%
#MCMC Chain Diagnostics 
#Use Arviz Package 
diag = az.summary(fit) #look at r_hat statistics
rhat_max = diag['r_hat'].max() #pass if <= 1.01 
#trace plots 
az.plot_trace(data = fit, var_names = ['~eta', '~eta_norm', '~sigma'], combined = False, compact = False)
az.bfmi(fit)

# %%
#Main Part 3: MFVB using KD's code 
var = ['nu.1', 'nu.2', 'nu.3', 'lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']  #hard coded order, not efficient: nu, lam, psi,sig2
mfvb = doMFVB()
#Returns dictionary with torch.distributions 
num_sample = torch.tensor([num_chains*num_samples])
mfvb_sample = np.concatenate([mfvb[key].rsample(num_sample).detach().numpy() for key in mfvb], axis = 1)
mfdf = pd.DataFrame(mfvb_sample, columns = var)

# %%
#MLE Estimation: not adjusted for dynamic M 
#Make y_data into a pandas dataframe
coln = ['y1', 'y2', 'y3']
data = pd.DataFrame(y_data.numpy(), columns = coln) 
desc = '''eta =~ y1 + y2 + y3'''
estimates = mle(data = data, desc = desc)
varnmle = ['lam_fixed','lam.1', 'lam.2','nu.1', 'nu.2', 'nu.3','sig2', 'psi.1', 'psi.2', 'psi.3']
mleest= dict(zip(varnmle, estimates['Estimate']))
# %%
#Save variables permanently (write to a file or something so I can use for later)
#Variables I want to save:
#pickle the y_data 

with open('y_datapickle.pickle','wb') as handle:
    pickle.dump(y_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

#pickle sem_model

with open('advi14071hpickle.pickle','wb') as handle:
    pickle.dump(sem_model, handle, protocol = pickle.HIGHEST_PROTOCOL)

#pickle mcmc 
with open('mcmc1407h1pickle.pickle', 'wb') as handle: 
    pickle.dump(fit, handle, protocol = pickle.HIGHEST_PROTOCOL)

# %%
#Comparison Plots

#1. Plot Densities 
data = {'MCMC': [mcdf, 'orange'],  'MFVB': [mfdf, 'green'], 'ADVI': [vbdf, 'blue']}
plot_dens(data = data, mle = mleest)

#2. Plot Eta
vbeta = sem_model.qvar['eta'].var_params[0].detach().numpy()
#extract mc eta 
filter_eta = [col for col in fitp if col.startswith('eta.')]
mceta = fitp[filter_eta].mean()
plot_etameans(vbeta = vbeta, mceta = mceta)

#3. Credible Intervals 
plot_credint(data = data)

# %%
