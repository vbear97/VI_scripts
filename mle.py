#%%
#Import Python specific packages
import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn

# %%
#Import semopy package for SEM 

from semopy import Model
from semopy.means import estimate_means
from semopy import ModelMeans
import pandas as pd
import numpy as np 
import pandas as pd
# # %%
# #Inherit data generating process (temporary use only)
# sig2_shape = torch.tensor([1.0])  
# sig2_rate = torch.tensor([1.0])  

# #psi ~ iid Inv Gamma for j = 1..m 
# psi_shape = torch.tensor([1.0])  
# psi_rate = torch.tensor([1.0])  

# #nu ~ iid Normal for j = 1...m
# nu_sig2 = torch.tensor([10.0])  
# nu_mean = torch.tensor([0.0])

# #lam_j | psi_j ~ id Normal(mu, sig2*psi_j)
# lam_mean = torch.tensor([1.0])
# lam_sig2 = torch.tensor([10.0])

# hyper = {"sig2_shape": sig2_shape, "sig2_rate": sig2_rate, "psi_shape": psi_shape, "psi_rate": psi_rate, "nu_sig2": nu_sig2, "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}

# # %% 
# #Set True Values for Parameters 
# N = 1000
# M = 3
# nu = torch.tensor([5.0, 10.0, 2.0])
# sig = torch.tensor([1.2])
# sig2= torch.square(sig)
# lam = torch.tensor([0.8, 0.5])
# psi_sqrt = torch.tensor([3.1, 2.2, 1.1])
# psi = torch.square(psi_sqrt)

# #Generate Latent Variables
# eta = torch.randn(N)*sig
# lam1 = torch.tensor([1.0])
# lam_full= torch.cat((lam1, lam))

# # %%
# #Generate y values based on User Input
# # yi ~ id Normal(nu + eta_i * lam, diag(psi)), yi /in R^m
# #cov:diag(psi)
# like_dist_cov = torch.diag(psi) #m*m tensor 
# #means: want a n*m vector of means
# like_dist_means = torch.matmul(eta.unsqueeze(1), lam_full.unsqueeze(0)) + nu
# #Generate yi
# y_data = mvn(like_dist_means, covariance_matrix= like_dist_cov).rsample() #n*m tensor
# # %%
# #Specify the sem model 
# #Step 1: Specify the sem model in string variable desc
# desc = '''eta =~ y1 + y2 + y3'''
# mod = Model(desc)
# #Step 2: Make y_data compatible
# coln = ['y1', 'y2', 'y3']
# data = pd.DataFrame(y_data.numpy(), columns = coln)
# #Invoke fit method of model instance for given data
# res = mod.fit(data, obj = "MLW")
# #invoke inspect method of model to analyse parameter estimates 
# estimates = mod.inspect()
# means = estimate_means(mod)

# mod2 = ModelMeans(desc)
# mod2.fit(data)
# estimate2=mod2.inspect()

# %%
#Write the function 

def mle(data, desc):
    #data = pandas dataframe of y1, y2, ...ym labelled
    #desc = model description
    mod = ModelMeans(desc)
    mod.fit(data)
    estimates = mod.inspect()
    return estimates 