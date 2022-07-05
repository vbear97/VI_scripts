#%%
#Import relevant torch packages 
import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn

#import loop packages 
from tqdm import trange

#import pystan packages 
import stan

#import relevant packages
import numpy
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# %%
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

hyper = {"sig2_shape": sig2_shape, "sig2_rate": sig2_rate, "psi_shape": psi_shape, "psi_rate": psi_rate, "nu_sig2": nu_sig2, "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}

# %% 
#Set True Values for Parameters 
N = 10
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
#Write mcmc function 

def mc(data):

    mccode= """
    data {
    int<lower=0> N; // number of individuals
    int<lower=0> K; // number of items
    vector[K] y[N]; //Y matrix of K items for N individuadls
    real lam_mean; //prior mean for lambda
    real<lower=0> lam_sig2; //prior variance for lambda
    real nu_mean; //prior mean for nu 
    real<lower=0> nu_sig2; //prior variance for nu 
    real<lower=0> sig2_shape; 
    real<lower=0> sig2_rate; 
    real<lower=0> psi_shape;  
    real<lower=0> psi_rate;
    }

    parameters {
    vector[N] eta_norm; // normalized eta for each individual
    vector[K] nu; // int for item k
    vector<lower=0>[K-1] lambday; // loading item m, fixing the first to be 1
    real <lower=0> sigma2; // var of the factor
    vector<lower=0>[K] psidiag; // sd of error
    }
    transformed parameters{
    vector[N] eta;
    real sigma;
    sigma = sqrt(sigma2);
    eta = sigma*eta_norm; 
    }

    model{
    vector[K] mu[N];
    matrix[K,K] Sigma;
    
    real cond_sd_lambda[K-1];
    vector[K] lambda;
    
    eta_norm ~ normal(0,1) ;
    lambda[1] = 1;
    lambda[2:K] = lambday;

    sigma2 ~ inv_gamma(sig2_shape, sig2_rate);
    
    for(k in 1:K){
        psidiag[k]~ inv_gamma(psi_shape, psi_rate);
        nu[k] ~ normal(0,sqrt(nu_sig2));    
    }
    
    for(k in 1:(K-1) ){
        cond_sd_lambda[k] = sqrt(lam_sig2*psidiag[k+1]);
        lambday[k] ~ normal(lam_mean,cond_sd_lambda[k]);
    }
    
    for(i in 1:N){   
        mu[i] = nu + lambda*eta[i];    
    }
    
    Sigma =  diag_matrix(psidiag);
    
    y ~ multi_normal(mu,Sigma); 
    }
        """
    #Build posterior 
    posterior = stan.build(mccode, data)
    #fit = posterior.sample(num_chains = num_chains, num_samples = num_samples) #do MCMC for default number of iterations, 
    #df = fit.to_frame()
    return posterior
# %%
#Prepare data dictionary

data = {"y": y_data.clone().numpy(),\
        "N": y_data.size(0),\
        "K": y_data.size(1)}
h = {var:param.item() for var,param in hyper.items()}
data.update(h)
#replace keys with appropriate names

#variable transform required for psi and sig2
#re-parameterise to inv-chi-squared distribution 

#change all data type from tensor to numpy 

#sample and visualise 

#don't need: 
#get rid of nu_mean, hardset to 0

# %%
#rough diagnostics: check the arithmetic mean, mode and variance 
