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

class sem_model():
    def __init__(self, y_data, degenerates, hyper):
        #inputs, as torch tensors:
        #y_data: (n*m) dataset
        #degenerates: dictionary with string keys from lam, nu,psi, eta, sig2: float tensor, degenerate
        #hyper: dictionary with string keys for prior distributions

        self.y_data = y_data
        self.n = y_data.size(0)
        self.m = y_data.size(1)
        self.degenerates = degenerates
        self.hyper = hyper
        #Create latent variable vector
        self.var_parameters = {"nu": q_var_normal(size=self.m), \
                          "lam": q_var_normal(size= self.m - 1),\
                          "eta": q_var_normal(size=self.n), \
                          "psi": q_var_normal(size=self.m), #to change, once inv-gamma function implemented
                          "sig2": q_var_normal(size=1)}       #to change, once inv-gamma function implemented

        for var in self.degenerates:
            self.var_parameters[var] = q_var_degenerate(values= self.degenerates[var]) #replace with degenerate distribution value

    def generate_theta_sample(self):
        #generate theta_sample with correct dimensions
        #output: key: torch.tensor dictionary
        theta_sample = {}
        for var in self.var_parameters:
            theta_sample[var] = self.var_parameters[var].rsample() #why do we need to do a for loop here? eliminate for loops
        #then, attach a 1 to lambda
        theta_sample['lam'] = torch.cat((torch.tensor([1]), theta_sample['lam']))
        return theta_sample

    def log_like(self, theta_sample):
        #theta_sample: full theta-sample from generate_theta_sample
        #returns: torch.tensor of log_likelihood

        nu = theta_sample['nu']  # tensor, size [m]
        lam = theta_sample['lam']  # tensor, size [m]
        eta = theta_sample['eta']  # tensor, size [n]

        like_dist_cov = torch.diag(theta_sample['psi'])  # m*m tensor

        # the best way like_dist_means (n*m tensor of means, i = nu + eta_i*lambda
        means = []
        for eta_i in eta:
            means.append(nu + eta_i * lam)
        like_dist_means = torch.stack(means, 0)

        like_dist = mvn(like_dist_means, covariance_matrix = like_dist_cov)
        log_likelihood_vec = like_dist.log_prob(self.y_data)
        log_likelihood_sum = log_likelihood_vec.sum()

        return log_likelihood_sum

    def log_prior(self, theta_sample):
        #if degenerate, don't add to the log_prior term
        to_add = {}
        for var in self.var_parameters:
            if var in self.degenerates:
                to_add[var] = False
            else:
                to_add[var] = True

        lp_sig2 = to_add['sig2'] * Inv_Gamma().log_prob(theta_sample['sig2']).sum()
        lp_psi = to_add['psi'] * Inv_Gamma().log_prob(theta_sample['psi']).sum()

        lp_nu = to_add['nu'] * (Normal(self.hyper['nu_mean'], torch.sqrt(self.hyper['nu_sig2'])).log_prob(theta_sample['nu']).sum())

        lp_lam_psi = to_add['lam'] * (Normal(self.hyper['lam_mean'], torch.sqrt(self.hyper['lam_sig2'] * theta_sample['psi'][1:])).log_prob(theta_sample['lam'][1:]).sum())

        lp_eta_sig2 = to_add['eta'] * (Normal(0, torch.sqrt(theta_sample['sig2'])).log_prob(theta_sample['eta']).sum())
        
        return lp_sig2 + lp_psi + lp_nu + lp_lam_psi + lp_eta_sig2

    def neg_entropy(self, theta_sample):
        neg_entropy = torch.tensor([0.0])
        for var in theta_sample:
            if var == "lam":
                neg_entropy += self.var_parameters[var].qvar_log(theta_sample[var][1:])
            else:
                neg_entropy += self.var_parameters[var].qvar_log(theta_sample[var]) #is there an occasion to delete the for loop?
        return neg_entropy

class q_var_normal():
    def __init__(self, size, mu=None, log_s=None):
        if mu is None:
            mu = torch.randn(size)
        if log_s is None:
            log_s = torch.randn(size)  # log of the standard deviation
        # Variational parameters
        self.phi = torch.stack([mu, log_s])  # stack rows on top of each other, note that log_s is allowed to be negative.
        self.phi.requires_grad = True
        self.size = size
    def dist(self):
        return torch.distributions.Normal(self.phi[0], self.phi[1].exp())
    def rsample(self, n=torch.Size([])):
        return self.dist().rsample(n)
    def qvar_log(self, real_num):
        return self.dist().log_prob(real_num).sum()

class q_var_degenerate():
    def __init__(self, values):
        self.phi = values
        self.phi.requires_grad = False #do not update this parameter
    def dist(self):
        return "Degenerate, see values attribute"
    def rsample(self):
        return self.phi.clone()
    def qvar_log(self, real_num):
        return torch.tensor([0.0])

class Inv_Gamma(): #dummy class
    def __init__(self):
        return #do nothing
    def log_prob(self, x):
        return torch.tensor([0.0])
###################################
def elbo(stats_model):
    theta_sample = stats_model.generate_theta_sample() #dictionary key:tensor
    return stats_model.log_like(theta_sample) + stats_model.log_prior(theta_sample) - stats_model.neg_entropy(theta_sample)

###################################
sig2_shape = torch.tensor([1.0])  # doesn't matter if degenerate
sig2_scale = torch.tensor([1.0])  # doesn't matter if degenerate
psi_shape = torch.tensor([1.0])  # doesn't matter if degenerate
psi_scale = torch.tensor([1.0])  # Doesn't matter if degenerate
nu_sig2 = torch.tensor([10.0])  # Normal
nu_mean = torch.tensor([0.0])

lam_mean = torch.tensor([1.0])  # Conditional Normal
lam_sig2 = torch.tensor([10.0])

hyper = {"sig2_shape": sig2_shape, "sig2_scale": sig2_scale, "psi_shape": psi_shape, "psi_scale": psi_scale,
         "nu_sig2": nu_sig2, \
         "nu_mean": nu_mean, "lam_mean": lam_mean, "lam_sig2": lam_sig2}

# Set true values for parameters
N = 1000
M = 3
nu = torch.tensor([5.0, 10.0, 2.0])
sig2 = torch.tensor([1.2 * 1.2])
lam = torch.tensor([0.8, 0.5])
psi_sqrt = torch.tensor([3.1, 2.2, 1.1])
psi = torch.square(psi_sqrt)
eta = torch.rand(N) * torch.sqrt(sig2)
params = {'nu': nu, \
          'eta': eta, 'psi': psi, \
          'lam': lam, 'sig2': sig2}

degenerates = {'psi': params['psi'], 'sig2': params['sig2'], 'eta': params['eta']}

#####################################
#Data Generating Process
#Generate y_data based on likelihood function
like_dist_cov = torch.diag(params['psi'])  # m*m tensor
nu = params['nu']
eta = params['eta']
lam = torch.cat((torch.tensor([1.0]), params['lam']))
means = []
for eta_i in eta:
    means.append(nu + eta_i * lam)
like_dist_means = torch.stack(means, 0)
like_dist = mvn(like_dist_means, covariance_matrix =like_dist_cov)
y_data = like_dist.sample()

#save data to csv file
#y_data_np = y_data.numpy()
#y_data_df = pd.DataFrame(y_data_np)
#y_data_df.to_csv('sem_y_data2.csv')

##########################################################
#Instantiate SEM model
sem_model = sem_model(y_data = y_data, degenerates = degenerates, hyper=hyper)

#Set Optimisation Parameters
lr, max_iter = 0.01, 5000
optimizer = torch.optim.Adam([sem_model.var_parameters[key].phi for key in sem_model.var_parameters], lr=lr)
iters = trange(max_iter, mininterval = 1)

#Optimisation Record Keeping

writer = SummaryWriter("semjune18_20")

#Optimise

for t in iters:
    optimizer.zero_grad()
    loss = -elbo(sem_model) #forward pass #maximise elbo, minimise loss
    loss.backward()
    optimizer.step()
    writer.add_scalar(tag = "training_loss: step_size="+str(lr), scalar_value=\
                      loss.item(), global_step = t)

    writer.add_scalars("vp",\
                       {'nu1_sig': sem_model.var_parameters['nu'].phi[1][0].exp().item(),\
                        'nu2_sig': sem_model.var_parameters['nu'].phi[1][1].exp().item(), \
                        'nu3_sig': sem_model.var_parameters['nu'].phi[1][2].exp().item(),\
                        'nu1_mean': sem_model.var_parameters['nu'].phi[0][0].item(),\
                        'nu2_mean': sem_model.var_parameters['nu'].phi[0][1].item(), \
                        'nu3_mean ': sem_model.var_parameters['nu'].phi[0][2].item(), \
                        'lambda2_mean': sem_model.var_parameters['lam'].phi[0][0].item(),\
                        'lambda2_sig': sem_model.var_parameters['lam'].phi[1][0].exp().item(),\
                        'lambda3_sig': sem_model.var_parameters['lam'].phi[1][1].exp().item(),\
                        'lambda3_mean': sem_model.var_parameters['lam'].phi[0][1].item(),\
                        'psi_1': sem_model.var_parameters['psi'].phi[0].item(),\
                        'psi_2': sem_model.var_parameters['psi'].phi[1].item(), \
                        'psi_3': sem_model.var_parameters['psi'].phi[2].item(),\
                    'sig2': sem_model.var_parameters['sig2'].phi[0].item(),\
                    }, global_step = t)
    # writer.add_scalars("eta", \
    #                    {'eta1_mean': sem_model.var_parameters['eta'].phi[0][0].exp().item(),\
    #                     'eta1_sig': sem_model.var_parameters['eta'].phi[1][0].exp().item(),\
    #                     'eta1_true': params['eta'][0].item(),\
    #                     'eta100_mean': sem_model.var_parameters['eta'].phi[0][99].exp().item(),\
    #                     'eta100_sig': sem_model.var_parameters['eta'].phi[1][99].exp().item(),\
    #                     'eta1_true': params['eta'][99].item(),\
    #                     'eta500_mean': sem_model.var_parameters['eta'].phi[0][500].exp().item(),\
    #                     'eta500_sig': sem_model.var_parameters['eta'].phi[1][500].exp().item(),\
    #                     'eta500_true': params['eta'][500].item(),\
    #                     'eta750_mean': sem_model.var_parameters['eta'].phi[0][750].exp().item(),\
    #                     'eta750_sig': sem_model.var_parameters['eta'].phi[1][750].exp().item(),\
    #                     'eta750_true': params['eta'][750].item(),\
    #                     }, global_step = t)

