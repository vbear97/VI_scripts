#Package to do with data visualisations 
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='The red data')
blue_patch = mpatches.Patch(color='blue', label='The blue data')

# %%
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


# %%
