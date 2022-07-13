
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

#Set MC Params
num_chains = 4
num_warmup = 7500
num_samples = 15000

lr_nl= 0.001
lr_ps= 0.01
lr_eta = 0.01
#psi and sigma are very slow to converge 

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

    # writer.add_scalars("eta", \
    #                    {'eta1_mean': sem_model.qvar['eta'].var_params[0][0].item(),\
    #                     'eta1_sig': sem_model.qvar['eta'].var_params[1][0].exp().item(),\
    #                     'eta1_true': eta[0].item(),\
    #                     'eta50_mean': sem_model.qvar['eta'].var_params[0][50].item(),\
    #                     'eta50_sig': sem_model.qvar['eta'].var_params[1][50].exp().item(),\
    #                     'eta50_true': eta[50].item(),\
    #                     'eta100_mean': sem_model.qvar['eta'].var_params[0][99].item(),\
    #                     'eta100_sig': sem_model.qvar['eta'].var_params[1][99].exp().item(),\
    #                     'eta100_true': eta[99].item(),\
    #                     'eta200_mean': sem_model.qvar['eta'].var_params[0][200].item(),\
    #                     'eta200_sig': sem_model.qvar['eta'].var_params[1][200].exp().item(),\
    #                     'eta200_true': eta[200].item(),\
    #                     'eta300_mean': sem_model.qvar['eta'].var_params[0][300].item(),\
    #                     'eta300_sig': sem_model.qvar['eta'].var_params[1][300].exp().item(),\
    #                     'eta300_true': eta[300].item(),\
    #                     'eta400_mean': sem_model.qvar['eta'].var_params[0][400].item(),\
    #                     'eta400_sig': sem_model.qvar['eta'].var_params[1][400].exp().item(),\
    #                     'eta400_true': eta[400].item(),\
    #                     'eta500_mean': sem_model.qvar['eta'].var_params[0][500].item(),\
    #                     'eta500_sig': sem_model.qvar['eta'].var_params[1][500].exp().item(),\
    #                     'eta500_true': eta[500].item(),\
    #                     'eta600_mean': sem_model.qvar['eta'].var_params[0][600].item(),\
    #                     'eta600_sig': sem_model.qvar['eta'].var_params[1][600].exp().item(),\
    #                     'eta600_true': eta[600].item(),\
    #                     'eta750_mean': sem_model.qvar['eta'].var_params[0][750].item(),\
    #                     'eta750_sig': sem_model.qvar['eta'].var_params[1][750].exp().item(),\
    #                     'eta750_true': eta[750].item(),\
    #                     }, global_step = t)
# %%
# %%
#Prepare data for MCMC
data = {"y": y_data.clone().numpy(),\
        "N": y_data.size(0),\
        "M": y_data.size(1)}
h = {var:param.item() for var,param in hyper.items()}
data.update(h)
# %%
#Initialise dispersed starting values for chains to maximise chance of detecting evidence of non-convergence 

# chain1init={'lam': [0,0]}
# chain2init={'lam': [1,1]}
# chain3init={'lam': [-1, -1]}
# chain4init={'lam': [0.5, 0.5]}
# init = [chain1init, chain2init, chain3init, chain4init]
#Not very efficient, make more flexible later

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
#Personally I find it easier just to look at the posterior histograms to see if they are all overlapping 

#Divergence, Tree Depth, E-BFMI ---> to do with model configuration, so not relevant 

#ESS is to do with efficiency of the sampler --> e.g. jumping distribution. but notice that for HMC the jumping distribution (normal) is fixed, we always accept. so maybe not relevant here?

#az.bfmi(data = fit)
#az.rhat(data = fit)
#az.mcse(data = fit)
#az.ess(data = fit)

# %%
#Main Part 3: MFVB using KD's code 



# %%
#Sample ADVI data 
#Sample VB data excluding eta ---< pd.df
num_sample = torch.tensor([num_chains * num_samples])
vb_sample = np.concatenate([sem_model.qvar[key].dist().rsample(num_sample).detach().numpy() for key in sem_model.qvar if key!= 'eta'], axis = 1)
vbdf = pd.DataFrame(vb_sample, columns = var)

# %%
#Results Analysis
#Test 1: Is MCMC mean close  to VB means?

#Test 1a.
mcdf.mean()
vbdf.mean()

#Test 2: MCMC variance vs. VB variances 
mcdf.var()
vbdf.var()
#How to compare sample variances
#Even if distribution does not have a second moment?


#Test 3: Numerical measures of comparison?
#Test 3.1 KD's Accuracy Measure. 
#Is that supposed to be squared euclidean distance?
acc = {}

# %%
#MLE Estimation: not adjusted for dynamic M 
#Make y_data into a pandas dataframe
coln = ['y1', 'y2', 'y3']
data = pd.DataFrame(y_data.numpy(), columns = coln) 
desc = '''eta =~ y1 + y2 + y3'''
estimates = mle(data = data, desc = desc)
varnmle = ['lam_fixed','lam.1', 'lam.2','nu.1', 'nu.2', 'nu.3','sig2', 'psi.1', 'psi.3', 'psi.2']
mleest= dict(zip(varnmle, estimates['Estimate']))
# %%
#Save variables permanently (write to a file or something so I can use for later)
#Variables I want to save:

#pickle the y_data 

with open('y_datapickle.pickle','wb') as handle:
    pickle.dump(y_data, handle, protocol = pickle.HIGHEST_PROTOCOL)

#pickle sem_model

with open('advi13071hpickle.pickle','wb') as handle:
    pickle.dump(sem_model, handle, protocol = pickle.HIGHEST_PROTOCOL)

#pickle mcmc 
with open('mcmc1307h1pickle.pickle', 'wb') as handle: 
    pickle.dump(fit, handle, protocol = pickle.HIGHEST_PROTOCOL)

# %%
#Plot Excluding Eta 
fig, ax = plt.subplots(5,2, constrained_layout = True, figsize = (10,10))#harded coded, not dynamic if we change the size of M
fig.delaxes(ax[4,1])
fig.suptitle("Estimated Posterior Densities conditional on Holzinger'39 Data")

#manually add in legend
or_patch = mpatches.Patch(color='orange', label='MCMC app post.')
blue_patch = mpatches.Patch(color='blue', label='ADVI app post.')
black_patch = mpatches.Patch(color = 'black', label = "MLE estimate" )

fig.legend(handles=[or_patch, blue_patch, black_patch], loc = 'lower right')

for v,a in zip(var,ax.flatten()):
    sns.histplot(data = mcdf[v], ax = a, color = 'orange', stat = 'density', kde = True) #mcmc density
    sns.histplot(data = vbdf[v], ax = a, stat = 'density', color = 'blue', bins = 100, kde = True) #vb density
    a.axvline(x = mleest[v],  color = 'black') #mle line 
    #print out accuracy measure from acc dictionary

etafig, etaax = plt.subplots(figsize = (5,5))
etafig.suptitle("Scatterplot comparing eta means")
vbeta = sem_model.qvar['eta'].var_params[0].detach().numpy()
#extract mc eta 
filter_eta = [col for col in fitp if col.startswith('eta.')]
mceta = fitp[filter_eta].mean()
etaax.scatter(x = mceta, y = vbeta)
etaax.set_xlabel('MCMC Eta Means')
etaax.set_ylabel('VB Eta Means')
etaax.axline(xy1 = (0,0), slope = 1)
# %%
#Plot eta for quality assurance properties
etafig, etaax = plt.subplots(figsize = (5,5)) 
etafig.suptitle('Scatterplot to check quality of eta approximation for VB')
vbeta = sem_model.qvar['eta'].var_params[0].detach().numpy()
etaax.scatter(x = eta.numpy(), y = vbeta)
etaax.set_xlabel("True Eta")
etaax.set_ylabel("VB Eta")
etaax.axline(xy1 = (0,0), slope = 1)

# %%
