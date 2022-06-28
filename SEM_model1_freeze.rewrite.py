
#%%
#Import relevant packages 
from re import S
from turtle import update
from sklearn.cluster import k_means

import torch
from torch.distributions import Normal, Gamma, Binomial
from torch.distributions import MultivariateNormal as mvn
from torch.utils.tensorboard import SummaryWriter

from sem import *
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
lam1 = torch.tensor([1.0])
lam_full= torch.cat((lam1, lam))

#Set Optim Params
lr, max_iter = 0.01, 20000
writer = SummaryWriter("test")

#Fix degenerates
degenerate = {'psi': psi, 'sig2': sig2, 'nu': nu, 'lam': lam} #degenerate lam is 2 dimensional

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
#Instantiate Optimizer Object
optimizer = torch.optim.Adam([sem_model.qvar[key].var_params for key in sem_model.qvar], lr = lr)

# %%
#Optimise
for t in range(max_iter):
    optimizer.zero_grad()
    loss = -sem_model.elbo()
    loss.backward()
    optimizer.step()

    print("psi_grad", sem_model.qvar['psi'].var_params.grad)
    print("nu_grad", sem_model.qvar['nu'].var_params.grad)
    print("lam_grad", sem_model.qvar['lam'].var_params.grad)


    writer.add_scalar(tag = "training_loss: step_size="+str(lr), scalar_value=\
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
                    # 'sig2_alpha': sem_model.qvar['sig2'].var_params[0].exp().item(),\
                    # 'sig2_beta': sem_model.qvar['sig2'].var_params[1].exp().item(),\
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