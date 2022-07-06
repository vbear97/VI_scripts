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

#Sample MC data
mcdf = fitp[var]

#Sample VB data (also a dataframe)
num_sample = torch.tensor([10000])
vb_sample = np.concatenate([sem_model.qvar[key].dist.rsample(num_sample).detach.numpy() for key in sem_model.qvar if key!= 'eta'], axis = 1)
vb_sample = pd.DataFrame(vb_sample, columns = var)



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

