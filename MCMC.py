#%%
#Import relevant torch packages 
import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn

#import loop packages 
from tqdm import trange

#import pystan packages 
import stan

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
#Task 1: Prepare MCMC code 

def mc(y_data, hyper, num_chains, num_samples):

    #return pandas dataframe of sampled values from mc posteriors, full model

    mc_code = """
    data{int<lower=1> N;
        int <lower=1> M; 
        matrix[N,M] real y; //y_data 
        real<lower=0> sig2_shape; 
        real <lower=0> sig2_rate; 
        real<lower=0> psi_shape;  
        real<lower=0> psi_rate; 
        real<lower=0> nu_sig2; 
        real nu_mean; 
        real lam_mean; 
        real<lower=0> lam_sig2;
    }
    parameters{
        vector[M] real nu; //intercept term
        vector[N] real eta; //latent 
        vector[M] real<lower=0> psi; //variance
        real<lower=0> sig2; //latent variance
        vector[M-1] real lam; //scaling term
    }
    transformed parameters{
        vector[M] lamf; 
        lamf = append_row(1, lam); //full lam vector 
        psi_cut = psi[2:]
    }

    model{
        array[N] vector[M] mu;

        for (n in 1:N) {
        mu[n] = lamf * eta[n] + nu
        }

        y ~ multi_normal(mu, diag_matrix(psi)); //likelihood definition
        sig2 ~ inv_gamma(sig2_shape, sig2_rate); //prior on sig2
        eta ~ normal(0, sqrt(sig2)); //prior on eta
        nu ~ normal(0, sqrt(nu_sig2)); //prior on nu
        psi ~ inv_gamma(psi_shape, psi_rate); //prior on psi
        lam ~ normal(lm, sqrt(lam_sig2 * psi_cut)); //prior on lam
    }
    """
    data = {"y": y_data,\
            "N": y_data.size(0),\
            "M": y_data.size(1),\
            }

    data.update(hyper)

    posterior = stan.build(mc_code, data)
    fit =posterior.sample(num_chains = num_chains, num_samples = num_samples)
    
    df = fit.to_frame()

    return df 