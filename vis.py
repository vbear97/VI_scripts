#Package to do with data visualisations 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
fig,ax = plt.subplots()
ax.set_title("hello")
ax.plot([1,2,3,4], [1,2,3,4]
# %%
#Make var string ##optimise later 
var = ['nu.1', 'nu.2', 'nu.3', 'lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']  #hard coded order: nu, lam, psi,sig2

#Sample MC data --> pd.df
mcdf = fitp[var]

#Sample VB data ---< pd.df
num_sample = torch.tensor([10000])
vb_sample = np.concatenate([sem_model.qvar[key].dist.rsample(num_sample).detach.numpy() for key in sem_model.qvar if key!= 'eta'], axis = 1)
vb_sample = pd.DataFrame(vb_sample, columns = var)




#Relevant code just for one plot 
fig, ax = plt.subplots()
sns.histplot(data = fitp['nu.1'], )

