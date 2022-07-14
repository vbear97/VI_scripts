#%%
#This is a file that manages visualisations 
# %%
#the main data source is a pandas dataframe with columns 
#nu.1, nu.2, lam.1, lam.2, psi.1, psi.2, psi.3

# %%
#For Visualisation and Sampling 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
# %%
def plot_dens(data, mle = None, title = 'Estimated Posterior Densities', figsize = (10,10), yeskde = True):
    var = ['nu.1', 'nu.2', 'nu.3','lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']
    #draw figure and axes 
    fig, ax = plt.subplots(5,2, constrained_layout = True, figsize = figsize)#harded coded, not dynamic if we change the size of M
    fig.delaxes(ax[4,1])
    fig.suptitle(title)

    #autocreate handles for legend
    handles = [mpatches.Patch(color = data[key][1], label = key + ' app post.') for key in data]
    fig.legend(handles= handles, loc = 'lower right')

    #draw plots
    for v,a in zip(var,ax.flatten()):
        if mle is not None:
            a.axvline(x = mle[v], color = 'black')
        if yeskde:
            for key in data:
                sns.kdeplot(data = data[key][0][v], color = data[key][1], ax = a)
        else:
            for key in data:
                sns.kdeplot(data = data[key][0][v], color = data[key][1], ax = a, stat = 'density', kde = True, bins = 100)
    
# %%
def plot_etameans(vbeta, mceta, ylabel = 'VB Eta Means', figsize = (5,5)):
    '''vbeta = single numpy array of eta means
    mceta = single numpy array of mc means
    '''
    fig, ax = plt.subplots(figsize = figsize)
    fig.suptitle("Scatterplot comparing eta means")
    #extract mc eta 
    ax.scatter(x = mceta, y = vbeta)
    ax.set_xlabel('MCMC Eta Means')
    ax.set_ylabel(ylabel)
    ax.axline(xy1 = (0,0), slope = 1)
# %%
def plot_credint(data, q1 = 0.0275, q2 = 0.975, figsize = (10,10), title = 'Credible Intervals'):
        '''
        data = {'MCMC': [mcdf, 'color' form}
        '''
        var = ['nu.1', 'nu.2', 'nu.3','lam.1', 'lam.2', 'psi.1', 'psi.2', 'psi.3', 'sig2']
        fig, ax = plt.subplots(5,2, constrained_layout = True, figsize = figsize)#harded coded, not dynamic if we change the size of M
        fig.delaxes(ax[4,1])
        fig.suptitle(title)
    
        #autocreate handles for legend
        handles = [mpatches.Patch(color = data[key][1], label = key + ' Cred Interval.') for key in data]
        handles.append(mpatches.Patch(color = 'red', ls = '--', label = 'MCMC Mean')) #MCMC mean

        fig.legend(handles= handles, loc = 'lower right')

        #extract quantiles
        quantiles = {key: [data[key][0].quantile([q1, q2]), data[key][0].mean()] for key in data}

        #plot as credible intervals
        for v,a in zip(var,ax.flatten()):
            for key, y in zip(data, range(len(data))):
                #plot credible interval
                color = data[key][1]
                a.plot(quantiles[key][0][v], (y,y), color = color, linewidth = 10, alpha = 0.5)
                #plot mean
                a.plot(x = quantiles[key][1][v], y = y, color = color, marker = 'o')

            #then, plot mcmc mean as a reference line 
            a.axvline(x = quantiles['MCMC'][v], color = 'red', ls = '--')
# %%
   # writer.add_scalars("eta", \
    #                    {'eta1_mean': sem_model.qvar['eta'].var_params[0][0].item(),\
    #                     'eta1_sig': sem_model.qvar['eta'].var_params[1][0].exp().item(),\
    #                     'eta1_true': eta[0].item(),\
    #                     'eta50_mean': sem_model.qvar['eta'].var_params[0][50].item(),\
    #                     'eta50_sig': sem_model.qvar['eta'].var_params[1][50].exp().item(),\
    #                     'eta50_true': eta[50].item(),\
    #                     'eta100_mean': sem_model.qvar['eta'].var_params[0][99].item(),\
    #                     'eta100_sig': sem_model.qvar['eta'].var_params[1][99].exp().item(),\
    #                     'eta100_true': eta[99].item(),\
    #                     'eta200_mean': sem_model.qvar['eta'].var_params[0][200].item(),\
    #                     'eta200_sig': sem_model.qvar['eta'].var_params[1][200].exp().item(),\
    #                     'eta200_true': eta[200].item(),\
    #                     'eta300_mean': sem_model.qvar['eta'].var_params[0][300].item(),\
    #                     'eta300_sig': sem_model.qvar['eta'].var_params[1][300].exp().item(),\
    #                     'eta300_true': eta[300].item(),\
    #                     'eta400_mean': sem_model.qvar['eta'].var_params[0][400].item(),\
    #                     'eta400_sig': sem_model.qvar['eta'].var_params[1][400].exp().item(),\
    #                     'eta400_true': eta[400].item(),\
    #                     'eta500_mean': sem_model.qvar['eta'].var_params[0][500].item(),\
    #                     'eta500_sig': sem_model.qvar['eta'].var_params[1][500].exp().item(),\
    #                     'eta500_true': eta[500].item(),\
    #                     'eta600_mean': sem_model.qvar['eta'].var_params[0][600].item(),\
    #                     'eta600_sig': sem_model.qvar['eta'].var_params[1][600].exp().item(),\
    #                     'eta600_true': eta[600].item(),\
    #                     'eta750_mean': sem_model.qvar['eta'].var_params[0][750].item(),\
    #                     'eta750_sig': sem_model.qvar['eta'].var_params[1][750].exp().item(),\
    #                     'eta750_true': eta[750].item(),\
    #                     }, global_step = t)
