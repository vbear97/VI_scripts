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
var = ['nu.1', 'nu.2', 'nu.3', 'lambday.1', 'lambday.2', 'psidiag.1', 'psidiag.2', 'psidiag.3', 'sigma2'] 


#Data for MCMC for non-latent 
var = ['nu.1', 'nu.2', 'nu.3', 'lambday.1', 'lambday.2', 'psidiag.1', 'psidiag.2', 'psidiag.3', 'sigma2']
mcdf = fitp[var]


#Data from VB for non-latent
num_sample = torch.tensor([10000])
nus = sem_model.qvar['nu'].dist().rsample(num_sample).detach().numpy()
vbdf= pd.DataFrame({'nu.1':nus[:,0]})
vbdf['nu2']= nu.2.tolist()
nu.3 = pd.DataFrame({'nu.3',nus[:,2]})
lamadas= sem_model.qvar['lam'].dist().rsample(num_sample).detach().numpy()
lamday.1=pd.DataFrame({'lamday.1':lamdas[:,0]})
lamday.2=pd.DataFrame({'lamday.2':lambdas[:,1]})
psis= sem_model.qvar['psi'].dist().rsample(num_sample).detach().numpy()
psidiag.1 = pd.DataFrame({'psidiag.1':psis[:,0]})
psidiag.2 = pd.DataFrame({'psidiag.2':psis[:,1]})
psidiag.3 = pd.DataFrame({'psidiag.3':psis[:,2]})
sigma2 = sem_pmodel.qvar['sig2'].dist().rsample(num_sample).detach().numpy()



#Relevant code just for one plot 
fig, ax = plt.subplots()
sns.histplot(data = fitp['nu.1'], )

