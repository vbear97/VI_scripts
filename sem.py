from re import S
from turtle import update
from sklearn.cluster import k_means

import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn
import pyro
from pyro.distributions import InverseGamma
from torch.utils.tensorboard import SummaryWriter

#qvar Distributions
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

class qvar_invgamma():
    def __init__(self, size, alpha=None, beta=None):
        if alpha is None:
            log_a = torch.randn(size) #log_alpha
        if beta is None:
            log_b = torch.randn(size) #log_beta
        # Variational parameters
        self.var_params = torch.stack([log_a, log_b]) #unconstrained
        self.var_params.requires_grad = True
    def dist(self):
        return InverseGamma(concentration= torch.exp(self.var_params[0]), rate = torch.exp(self.var_params[1]))
    def rsample(self, n = torch.Size([])):
        return self.dist().rsample(n)
    def log_prob(self,x):
        return self.dist().log_prob(x).sum() #assume independent components

# class qvar_invgamma(): #dummy class
#     def __init__(self):
#         self.var_params = None 
#         return #do nothing
#     def log_prob(self, x):
#         return torch.tensor([0.0])

# class InvGamma(): #dummy class 
#     def __init__(self):
#         return #do nothing
#     def log_prob(self, x):
#         return torch.tensor([0.0])

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
