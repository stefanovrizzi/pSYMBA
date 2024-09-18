import os

import numpy as np
import argparse

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
from scipy.interpolate import griddata

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

import string

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching

from plot.palette_colorblind import palette, conditions_order, dict_string2num, dict_num2label, dict_label2num, biasDict

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial']})
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

import seaborn as sns

class Plot:

    def __init__(self, n):
        
        store = FileStorage()
        fetch = FileFetching(store)
        comparison, n = fetch.comparison(n)
        
        ComparisonFolder = fetch.plotFolderPath+f'/comparison_{n}'

        self.exStart = comparison['exStart']
            
        CognitiveTraits = comparison['cognitiveTraits']        
        randomSeeds = comparison['randomSeeds']        
        C = len(CognitiveTraits)
        R = len(randomSeeds)

        self.N = R*C
        
        datasets = range(self.exStart, self.exStart+self.N)

        store = FileStorage()
        store.set_example(self.exStart)
        fetch = FileFetching(store)

        self.setup(self.exStart)
        self.create_folders()
        
        self.order = conditions_order
        self.palette=palette
        self.biasDict = biasDict
        self.dict_string2num = dict_string2num
        
    def setup(self, exN):
    
        store = FileStorage()
        store.set_example(exN)
        self.fetch = FileFetching(store)

        self.InputParameters = self.fetch.input_parameters()
        self.InputTimeParameters = self.fetch.input_parameters(time=True)
        self.input_parameters()

    def create_folders(self):

        #Create folders
        os.mkdir(self.fetch.plotFolderPath) if not os.path.isdir(self.fetch.plotFolderPath) else None #create Data folder for all data files, in case it does not exist
        
        os.mkdir(self.fetch.plotFolderPath+self.fetch.example_folder_path()) if not os.path.isdir(self.fetch.plotFolderPath+self.fetch.example_folder_path()) else None
        
        self.plotPath = self.fetch.plotFolderPath+f'/comparison_{n}'
        os.mkdir(self.plotPath) if not os.path.isdir(self.plotPath) else None #create Data folder for all data files, in case it does not exist

        self.plotPath += '/Figures'
        os.mkdir(self.plotPath) if not os.path.isdir(self.plotPath) else None #create Data folder for all data files, in case it does not exist

    def input_parameters(self):
    
        #Input parameters
        self.Ttraining = self.InputTimeParameters['Ttraining']
        self.tInit = self.fetch.parameters_json('SYMBA')['tInit'] #-100
        self.Ttesting = self.InputTimeParameters['T']

        self.I = self.InputParameters['I']
        self.S = self.InputTimeParameters['S']
                
        self.Tw, self.Tm, self.Ty = self.fetch.time_references()
        
    # Price dynamics
    def market_dynamics(self):
    
        Price = np.zeros((self.S, self.Ttraining+self.Ttesting-self.tInit))
       
        for s in range(self.S):
       
            Price[s, :self.Ttraining] = self.fetch.saved('Price', s='training_0')[0]
            Price[s, self.Ttraining:] = self.fetch.saved('Price', s=f'{s}')[0, self.tInit:]        

        lw = 1

        fig, ax = plt.subplots(1, 1, figsize=(12,4))

        #Price dynamics
        time = range(-self.tInit, 0)
        plt.plot(time, Price[0, :self.tInit], color=[.9, .9, .9], linestyle='-', label='Price buffer', lw=lw)
        
        time = range(0, self.tInit)
        plt.plot(time, Price[0, self.tInit:self.tInit*2], color=[.6, 0, 0], label='Action buffer', lw=lw)
        
        time = range(self.tInit, self.Ttraining-self.tInit)
        plt.plot(time, Price[0, self.tInit*2:self.Ttraining], color=[10/250, 120/250, 194/250], label='Learning buffer', lw=lw)
        
        time = range(self.Ttraining-self.tInit, self.Ttraining+self.Ttesting-self.tInit-self.tInit)
        [ plt.plot(time, Price[s, self.Ttraining:], color='k', alpha=.5, lw=lw) for s in [4, 7] ]

        #plt.axvline(0, color=[.5, .5, .5], ls='--', lw=lw/5)  
        #plt.axvline(252, color=[.5, .5, .5], ls='--', lw=lw/5)        
        #plt.axvline(self.Ttraining-self.tInit, color=[.5, .5, .5], ls='--', lw=lw/5)
        
        ticks = [0,252,self.Ttraining-self.tInit]
        ax.set_xticks(ticks)

        dic = { 0 : r"$t_{0}$", 252 : r"$t_{\text{learn}}$", self.Ttraining-self.tInit : r"$T_{\text{training}}$" }
        labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
        ## or 
        # labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]

        ax.set_xticklabels(labels)

        plt.plot(time, Price[self.S-1, self.Ttraining:], color='k', label=r'Simulation $s$ (testing)', lw=lw)
        
        plt.ylabel('Stock price (€)')
        plt.xlabel('Trading day (t)')
        
        plt.legend(loc=2)
            
        title = f'fig_1_price.svg'
        plt.savefig(self.plotPath+title, bbox_inches='tight')        
        plt.close()    

    def scatter_plot_alphas(self, exN):
    
        # Create a figure with 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True)
    
        for n, ax in enumerate(axes):
            store = FileStorage()
            store.set_example(exN + n)
            fetch = FileFetching(store)
        
            # RL parameters
            RLparameters = fetch.parameters_csv('RL')
        
            alphaConf = RLparameters['alphaConf']
            alphaDisc = RLparameters['alphaDisc']
            
            # Learning condition
            input_params = fetch.input_parameters()
            condition = input_params['CognitiveTraits'][0]            
            condition_abbreviation = self.biasDict[condition]
            
            n_ = self.dict_string2num[condition]
        
            ax = axes.flat[n_]
        
            ax.scatter(alphaDisc, alphaConf, color='black', alpha=0.05)
            ax.plot([0, 0.5], [0, 0.5], color='gray', linestyle='--')
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 0.5)
            ax.set_xlabel(r'$\alpha_{-}$')
            ax.set_ylabel(r'$\alpha_{+}$') if n_ == 0 else None
            
            ax.text(-0.2, 1.0, string.ascii_lowercase[n_], transform=ax.transAxes, size=12, weight='bold') if n_ == 0 else ax.text(-0.1, 1.0, string.ascii_lowercase[n_], transform=ax.transAxes, size=12, weight='bold')
    
        plt.tight_layout()
    
        title = '/alpha_scatter_plots.pdf'
        plt.savefig(self.plotPath + title, dpi=300, bbox_inches='tight')
        plt.close()

    def price_dynamics(self, exN, s):
        # Normalization for the colormap
        cmap = mpl.colormaps['coolwarm_r'] #cm.get_cmap('coolwarm_r')
        norm = mcolors.Normalize(vmin=0, vmax=1)
    
        ylimmax = 0
    
        # Create figure and set layout
        fig, axes = plt.subplots(2, 2, figsize=(8, 4.5), sharex=True, sharey=True)

        # Line plots in a 2x2 grid
        for n, _ in enumerate(axes.flat):
            store = FileStorage()
            store.set_example(exN + n)
            fetch = FileFetching(store)    

            input_params = fetch.input_parameters()
            condition = input_params['CognitiveTraits'][0]
            condition_abbreviation = self.biasDict[condition]
            
            n_ = self.order.index(condition_abbreviation)
        
            ax = axes.flat[n_]
        
            Price = fetch.saved('Price', s=str(s))[0]
            fundamentalValueTrue = fetch.saved('fundamentalValueTrue', s=str(s))[0]
            actionForecasting = fetch.saved('actionForecasting', s=str(s))[:, 0]        
            aFmean = actionForecasting.mean(axis=0)
            
            ylimmax = max(Price.max(), ylimmax, fundamentalValueTrue.max())
    
            for t in range(len(Price) - 1):
                ax.plot([t, t + 1], [Price[t], Price[t + 1]], color=cmap(norm(aFmean[t])))
        
            ax.plot(fundamentalValueTrue, color='k', alpha=0.3, label=r'$f$')
        
            ax.set_xlabel('Trading day (t)') if n_>1 else None
            ax.set_ylabel('Stock price') if n_ % 2 == 0 else None
            
            ax.text(-0.20, 1.0, string.ascii_lowercase[n_], transform=ax.transAxes, size=12, weight='bold') if n_ in [0, 2] else ax.text(-0.075, 1.0, string.ascii_lowercase[n_], transform=ax.transAxes, size=12, weight='bold')
            
            ax.text(0.5, 1.0, condition_abbreviation, transform=ax.transAxes, size=12)

        plt.ylim([0, ylimmax*1.05])

        # Adjust layout
        fig.subplots_adjust(wspace=0.4)

        # Create the colorbar
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.046, pad=-0.25)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Trend-following', 'Fundamentalism'])
    
        plt.legend()
        plt.tight_layout()
    
        title = f'/price_plots_s{s}.pdf'
        plt.savefig(self.plotPath + title, dpi=300, bbox_inches='tight')
        plt.close()

    def price_dynamics_joint(self, exN):
    
        # Normalization for the colormap
        cmap = cm.get_cmap('coolwarm_r')
        norm = mcolors.Normalize(vmin=0, vmax=1)
    
        # Create figure and set layout
        fig, axes = plt.subplots(2, 4, figsize=(13, 6), gridspec_kw={'width_ratios': [1, 2, 1, 2]})

        # Remove empty third column by hiding axes
        #for ax in axes[:, 2]:
        #    ax.axis('off')

        # Scatter plots in first and third column
        for n, ax in enumerate([axes[0, 0], axes[1, 0], axes[0, 2], axes[1, 2]]):
            store = FileStorage()
            store.set_example(exN+n)
            fetch = FileFetching(store)
        
            # RL parameters
            RLparameters = fetch.parameters_csv('RL')
    
            alphaConf = RLparameters['alphaConf']
            alphaDisc = RLparameters['alphaDisc']
            
            print(fetch.input_parameters()['CognitiveTraits'])
            
            ax.scatter(alphaConf, alphaDisc, color='black', alpha=0.05)
            ax.plot([0, 0.5], [0, 0.5], color='gray', linestyle='--')
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 0.5)
            ax.set_xlabel(r'$\alpha_{-}$')
            ax.set_ylabel(r'$\alpha_{+}$')

        # Line plots in second and fourth column
        for n, ax in enumerate([axes[0, 1], axes[1, 1], axes[0, 3], axes[1, 3]]):
            store = FileStorage()
            store.set_example(exN+n)
            fetch = FileFetching(store)    
        
            Price = fetch.saved('Price', s='0')[0]
            fundamentalValueTrue = fetch.saved('fundamentalValueTrue', s='0')[0]
            actionForecasting = fetch.saved('actionForecasting', s='0')[:, 0]        
            aFmean = actionForecasting.mean(axis=0)
    
            for t in range(len(Price) - 1):
                ax.plot([t, t + 1], [Price[t], Price[t + 1]], color=cmap(norm(aFmean[t])))
        
            ax.plot(fundamentalValueTrue, color='k', alpha=0.3, label=r'$f$')
        
            ax.set_xlabel('Trading day (t)')
            ax.set_ylabel('Stock price')

        # Adjust layout to create space between second and third columns
        #fig.subplots_adjust(wspace=0.4)

        # Create the colorbar
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation='horizontal', fraction=0.046, pad=-0.25)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Trend-following', 'Fundamentalism'])

        # Ensure subplots on the same column share the same x-axis
        #for ax in [axes[0, 0], axes[1, 0]]:
        #    ax.get_shared_x_axes().joined(ax, axes[0, 0])
        #for ax in [axes[0, 1], axes[1, 1]]:
        #    ax.get_shared_x_axes().joined(ax, axes[0, 1])
        #for ax in [axes[0, 3], axes[1, 3]]:
        #    ax.get_shared_x_axes().joined(ax, axes[0, 3])
    
        plt.legend()
        
        plt.tight_layout()

        title = '/Price_fund_bias_joint.png'
        
        plt.savefig(self.plotPath+title, dpi=300)

    def phase_plot(self, exN):

        palette = ['r',          #CB                           
                  'b',           #PB
                  [.8, 0, .8],   #NB
                  [.3, 0, .3],   #NNB                     
                  [.5, .5, .5]]  #NL
    
        # Create figure and set layout
        plt.figure(figsize=(8, 8))

        # Line plots in a 2x2 grid
        for n in range(5):
            store = FileStorage()
            store.set_example(exN + n)
            fetch = FileFetching(store)    
        
            Price = fetch.saved('Price', s='0')[0]
            fundamentalValueTrue = fetch.saved('fundamentalValueTrue', s='0')[0]
            
            #x = Price - fundamentalValueTrue
            x = np.log(Price) - np.log(fundamentalValueTrue)
            #x = abs(x)
            #x = np.log10(x+0.01)
            
            actionForecasting = fetch.saved('actionForecasting', s='0')[:, 0]        
            nR = actionForecasting.mean(axis=0)
            
            #x, nR = x[::25], nR[::25] 
                        
            label = fetch.input_parameters()['CognitiveTraits'][0]
    
            #plt.plot(x, nR, color=palette[n], label=label, lw=.5)
            plt.plot(nR, x, color=palette[n], label=label, lw=.3, alpha=.4)
            plt.scatter(np.median(nR), np.median(x), color=palette[n], label=f'{label} median', marker='x', alpha=1)
        
        #plt.ylim([-1000, 1000])
        plt.ylabel(r'$\log{P}-\log{f}$')
        plt.xlabel(r'$\bar{a}_{\mathcal{F}}$')
                
        plt.legend()
        
        title = '/phase_plot.png'
        plt.savefig(self.plotPath + title, dpi=300, bbox_inches='tight')
        plt.close()

    def kirman_plot_grid(self, exN, N):
        cognitive_traits = ['optimistic bias', 'no bias', 'no bias individual', 'pessimistic bias', 'no learning']
        abbreviations = [r'$B_{+}$', r'$B_{0}$', r'$B_{00}$', r'$B_{-}$', r'$Z$']
        biasDict = dict(zip(cognitive_traits, abbreviations))
        
        # Define how many subplots (5 x 4 = 20 subplots)
        fig, axes = plt.subplots(5, 4, figsize=(10, 8), sharex=True, sharey=True)
        axes = axes.flatten()  # Flatten to easily iterate over
        colname = r'$\mu_{\mathcal{F}, I}$'  # Example column name, adjust as needed
        index = np.arange(0, 2500)  # Example index for data, adjust as needed

        df_list = []  # Placeholder for collecting data

        # Loop over N conditions in chunks of 5
        for i in range(0, N, 5):
            df_chunk_list = []  # Temporary list to hold data for the current chunk of 5

            for n in range(i, i + 5):  # Handle 5 "n" values at a time
                store = FileStorage()
                store.set_example(exN + n)
                fetch = FileFetching(store)

                condition = fetch.input_parameters()['CognitiveTraits'][0]
                InputTimeParameters = fetch.input_parameters(time=True)
                S = InputTimeParameters['S']

                if condition != 'no learning':
                    condition = biasDict[condition]

                    for s in range(S):
                        # Fetch the actionForecasting data
                        actionForecasting = fetch.saved('actionForecasting', s=f'{s}')[:, 0, self.tInit:]
                        actionForecasting = np.mean(actionForecasting, axis=0)

                        df_temp = pd.DataFrame(data=actionForecasting, columns=[colname], index=index)
                        df_temp['condition'] = condition
                        df_chunk_list.append(df_temp)  # Add data for this specific condition
            
            # Concatenate the current chunk of 5 into a single DataFrame
            df_chunk = pd.concat(df_chunk_list)
            df_chunk['condition'] = pd.Categorical(df_chunk['condition'], categories=self.order[:5], ordered=True)
            
            # Get the corresponding axis (subplot) for the current group of 5
            ax = axes[i // 5]

            # Plot the KDE plot for this group of 5
            sns.kdeplot(data=df_chunk, x=colname, hue='condition', ax=ax, palette=self.palette[:5], hue_order=self.order[:5], legend=False, bw_adjust=0.1)

            # Add alphabetical labels to the subplot
            letter = chr(97 + (i // 5))  # Alphabetical labeling (a, b, c, ...)
            ax.text(-0.1, 1.05, letter, transform=ax.transAxes, size=12, weight='bold')

        # Set common x and y labels
        fig.text(0.5, 0.04, r'$\mu_{\mathcal{F}, I}$', ha='center')
        
        # Adjust layout
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
        
        # Save the figure
        title = '/kirman_kde_plot_grid.pdf'
        plt.savefig(self.plotPath + title, dpi=300, bbox_inches='tight')
        plt.close()



    def kirman_plot(self, exN, N):

        cognitive_traits = ['optimistic bias', 'no bias', 'no bias individual', 'pessimistic bias', 'no learning']
        abbreviations = [r'$B_{+}$', r'$B_{0}$', r'$B_{00}$', r'$B_{-}$', r'$Z}$']
        biasDict = dict( zip(cognitive_traits, abbreviations) )

        store = FileStorage()
        store.set_example(exN)
        fetch = FileFetching(store)
        T = self.InputTimeParameters['T']
        tInit = fetch.parameters_json('SYMBA')['tInit'] #-100

        index = range(T-tInit)

        bins = np.arange(0.0, 1.05, .1)

        #C = 5 #4 #conditions

        labels = []
            
        # Create figure and set layout
        plt.figure(figsize=(6, 4))

        df_list = []
        colname = 'ColName' #r'$\bar{a_{\mathcal{F}}$'
        
        for n in range(N):
        
            store = FileStorage()
            store.set_example(exN+n)
            fetch = FileFetching(store)                

            condition = fetch.input_parameters()['CognitiveTraits'][0]
            InputTimeParameters = fetch.input_parameters(time=True)
            S = InputTimeParameters['S']
            
            if condition != 'no learning':
            
                condition_abbreviation = biasDict[condition]

                for s in range(S):
                    actionForecasting = fetch.saved('actionForecasting', s=f'{s}')[:, 0, tInit:]    
                    actionForecasting = np.mean(actionForecasting, axis=0) #mean over traders 

                    df_temp = pd.DataFrame(data=actionForecasting, columns=[colname], index=index)
                    df_temp['condition'] = condition_abbreviation
                    #df_temp['random_seed'] = fetch.input_parameters()['randomSeed']

                    df_list.append(df_temp)
        
        df = pd.concat(df_list)
   
        df['condition'] = pd.Categorical(df['condition'], categories=self.order[:4], ordered=True)
        df = df.sort_values(by='condition')
   
        #plt.hist(nR.T, label=labels, bins=bins, width=.01, density=True) #palette=palette[:C],
        sns.kdeplot(data=df, x=colname, hue='condition', palette=self.palette[:4], hue_order=self.order[:4], legend=True, bw_adjust=.01)
        
        #plt.ylim([-1000, 1000])
        plt.xlabel(r'$\mu_{\mathcal{F}, I}$')
        plt.xlim([0, 1])
        
        title = '/kirman_plot_median.pdf'
        plt.savefig(self.plotPath + title, dpi=300, bbox_inches='tight')
        plt.close()


        #store = FileStorage()
        #store.set_example(exN)
        #fetch = FileFetching(store)
        #T = self.InputTimeParameters['T']
        #tInit = self.fetch.parameters_json('SYMBA')['tInit'] #-100

        #nR = np.zeros((C, T-tInit))

        #for n in range(C):
        #    store = FileStorage()
        #    store.set_example(exN + n)
        #    fetch = FileFetching(store)    
                    
        #    actionForecasting = fetch.saved('actionForecasting', s='0')[:, 0, tInit:]        
        #    nR[n] = actionForecasting.mean(axis=0)
            
            #x, nR = x[::25], nR[::25] 
                        
        #    labels.append( fetch.input_parameters()['CognitiveTraits'][0] )
        

    def forecast_preference_old(self, exN):
        s = 0  # simulation testing n

        # Create figure for tau vs aFmean
        fig_tau, axes_tau = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    
        for i, ax in enumerate(axes_tau.flat):
            store = FileStorage()
            store.set_example(exN + i)  # Change example every subplot
            fetch = FileFetching(store)
        
            actionForecasting = fetch.saved('actionForecasting', s=f'{s}')[:, 0]
            aFmean = actionForecasting.mean(axis=1)
            y = aFmean
        
            parameters = fetch.parameters_csv('SYMBA')
            tau = parameters['tau']
        
            x = tau
            ax.scatter(x, y, alpha=0.2, color='k')
            ax.set_xlabel(r'$\tau$')
            
            # Fit and plot linear model
            slope, intercept, _, _, _ = stats.linregress(x, y)
            ax.plot(x, slope * x + intercept, color='c', linestyle='--', label='linear regression')
            
            self.loess(x, y, ax)
            
            ax.set_ylabel(r'$\bar{a}_{\mathcal{F}}$') if i % 2 == 0 else None
            ax.legend() if i == 1 else None
                
        # Adjust layout to create space between pairs of subplots
        fig_tau.subplots_adjust(wspace=0.3, hspace=0.4)
        plt.tight_layout()
    
        # Save the figure for tau
        title_tau = f'/Forecast_preference_tau_{s}.png'
        plt.savefig(self.plotPath + title_tau, dpi=300)
        plt.close(fig_tau)

        # Create figure for kappa vs aFmean
        fig_kappa, axes_kappa = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    
        for i, ax in enumerate(axes_kappa.flat):
            store = FileStorage()
            store.set_example(exN + i)  # Change example every subplot
            fetch = FileFetching(store)
        
            actionForecasting = fetch.saved('actionForecasting', s=f'{s}')[:, 0]
            aFmean = actionForecasting.mean(axis=1)
            y = aFmean
        
            parameters = fetch.parameters_csv('SYMBA')
            kappa = parameters['kappa']
        
            #x = np.log10(kappa)
            x = kappa
            ax.scatter(x, y, alpha=0.2, color='k')
            ax.set_xlabel(r'$\log_{10}{\kappa}$')
            
            # Fit and plot linear model (log scale for kappa)
            slope, intercept, _, _, _ = stats.linregress(x, y)
            ax.plot(x, slope * x + intercept, color='c', linestyle='--', label='linear regression')
            
            self.loess(x, y, ax)
            
            ax.set_ylabel(r'$\bar{a}_{\mathcal{F}}$') if i % 2 == 0 else None
            ax.legend() if i == 1 else None
    
        # Adjust layout to create space between pairs of subplots
        fig_kappa.subplots_adjust(wspace=0.3, hspace=0.4)
        plt.tight_layout()
    
        # Save the figure for kappa
        title_kappa = f'/Forecast_preference_kappa_{s}.png'
        plt.savefig(self.plotPath + title_kappa, dpi=300)
        plt.close(fig_kappa)


    def forecast_preference(self, exN, N):

        C = 4

        #cmap = 'coolwarm_r' #cm.get_cmap('coolwarm_r')
        cmap = 'seismic_r' #cm.get_cmap('coolwarm_r')

        ticks = [0, 0.5, 1]
        tickLabels = list(map(str, ticks))

        # Create figure for tau vs aFmean
        #fig, axes = plt.subplots( int(N/C), C, figsize=(5, 7), sharex=True, sharey=True)
        fig, axes = plt.subplots( 10, C, figsize=(5, 9), sharex=True, sharey=True)        
        axes = axes.flatten()
    
        i = -1
    
        for i_ in range(N):
        
            store = FileStorage()
            store.set_example(exN + i_)  # Change example every subplot
            fetch = FileFetching(store)

            condition = fetch.input_parameters()['CognitiveTraits'][0]
            
            if condition != 'no learning':

                i += 1
                
                SYMBAparameters = fetch.parameters_csv('SYMBA')
                tau = SYMBAparameters['tau']
                kappa = SYMBAparameters['kappa']
            
                SYMBAparameters = fetch.parameters_json('SYMBA')
                kappamin = SYMBAparameters['kappamin']
                kappamax = SYMBAparameters['kappamax']
                taumin = SYMBAparameters['taumin']
                taumax = SYMBAparameters['taumax']            
                         
                #x = np.log(tau.values)
                #x -= min(x)
                #x /= max(x)
                
                x = tau.values
                x = x.astype(float)
                x -= taumin
                x /= (taumax-taumin)                
                #x = tau.values/taumax

                y = kappa.values
                y -= kappamin
                y /= (kappamax-kappamin)
             
                #y = kappa.values / kappamax
                     
                #y = np.log(kappa.values)
                #y -= np.log(kappamin)
                #y /= (np.log(kappamax)-np.log(kappamin))
            
                resolution = 100
                x *=resolution #resolution
                y *=resolution
            
                res0x = int(taumin/taumax)
                res0y = 0 #int( np.log(kappamin) / (np.log(kappamax)-np.log(kappamin)))
            
                z = np.zeros(( self.S, len(x) ))
                X = np.zeros_like(z)
                Y = np.zeros_like(z)
        
                for s in range(self.S):

                    aFtemp = fetch.saved('actionForecasting', s=f'{s}')[:, 0]
                    z[s] = aFtemp[:, self.tInit:].mean(axis=1)
                    X[s] = x.copy()
                    Y[s] = y.copy()
            
                x = X.flatten()
                y = Y.flatten()
                z = z.flatten()
        
                grid_x, grid_y = np.mgrid[res0x:resolution+1, res0y:resolution+1]
                z_interpolated = griddata((x,y), z, (grid_x, grid_y), method='linear')

                hue_order_position = self.dict_string2num[condition] + (i // C)*C

                ax_temp = axes[hue_order_position]
                #letter = string.ascii_lowercase[hue_order_position]

                print(condition, hue_order_position) #letter, 

                #ax_temp.text(-0.115, 1.0, letter, transform=ax_temp.transAxes, size=12, weight='bold') if hue_order_position > 0 else ax_temp.text(-0.2, 1.0, letter, transform=ax_temp.transAxes, size=12, weight='bold')
                #breakpoint()
                im = ax_temp.imshow(z_interpolated, cmap=cmap, origin='lower', vmin=0, vmax=1, aspect='auto', extent=(0, 1, 0, 1))
                #ax.set_xlabel(r'$\log{\kappa}$ (normalised)')
                #ax.set_ylabel(r'$\tau$ (normalised)') if i == 0 else None            

                ax_temp.set_xticks(ticks)
                ax_temp.set_xticklabels(tickLabels) if i > 35 else None
                ax_temp.set_yticks(ticks)
                ax_temp.set_yticklabels(tickLabels) if i % C == 0 else None

                #if i == N-1:
                    #cbar = plt.colorbar(im)
                    #cbar.ax.set_title(r'$\bar{a}_{\mathcal{F}}$')                

                    #divider = make_axes_locatable(ax)
                    #cax = divider.append_axes('right', size='50%', pad = 0.1)
                    #cax.set_title(r'$\bar{a}_{\mathcal{F}}$')
                    #fig.colorbar(im, cax=cax, orientation='vertical')
        
            #cax = divider.append_axes('right', size='5%', pad = 0.1)
            #cax.set_title(r'$\bar{a}_{\mathcal{F}}$')

        fig.text(0.0, 0.5, r'$\tau$ (normalised)', va='center', rotation='vertical')        
        fig.text(0.5, 0.05, r'$\kappa$ (normalised)', ha='center')
        #fig.text(0.5, -0.01, r'$\kappa$ (normalised)', ha='center')
        #plt.xlabel(r'$\log{\kappa}$ (normalised)')
            
        # Adjust layout to create space between pairs of subplots
        #fig.subplots_adjust(wspace=0.3, hspace=0.4)
        #plt.tight_layout()

        fig.subplots_adjust(wspace=0.15, hspace=0.3)

        cbaxes = fig.add_axes([0.95, 0.135, 0.03, 0.7])
        fig.colorbar(im, cax=cbaxes)
        cbaxes.set_title(r'$\bar{a}_{\mathcal{F}}$')
    
        # Save the figure for kappa
        title = f'/Forecast_preference_mesh_kappalog.pdf'
        plt.savefig(self.plotPath + title, dpi=300, bbox_inches='tight')
        plt.close()

    def forecast_preference_avg_all(self, exN, N, log_norm=False):
        
        C = 5
        cmap = 'seismic_r'

        ticks = [0, 0.5, 1]
        tickLabels = list(map(str, ticks))

        # Create figure for tau vs aFmean
        fig, axes = plt.subplots( 1, C, figsize=(9, 2), sharex=True, sharey=True) #, constrained_layout=True)
        axes = axes.flatten()

        X = np.zeros((C, int(N/C), self.S, self.I))
        Y = np.zeros_like(X)
        Z = np.zeros_like(X)
        N_ = np.zeros(C)-1 #randomseed count per condition

        SYMBAparameters = self.fetch.parameters_json('SYMBA')
        kappamin = SYMBAparameters['kappamin']
        kappamax = SYMBAparameters['kappamax']
        taumin = SYMBAparameters['taumin'] #5 #SYMBAparameters['taumin']
        taumax = SYMBAparameters['taumax']  

        resolution = 1000  
        res0x = int(taumin/taumax*resolution)
        res0y = 0 #int( np.log(kappamin) / (np.log(kappamax)-np.log(kappamin)))
            
        for n in range(N):
             
            store = FileStorage()
            store.set_example(exN + n)  # Change example every subplot
            fetch = FileFetching(store)

            condition = fetch.input_parameters()['CognitiveTraits'][0]
            c = self.dict_string2num[condition]
                    
            N_[c] += 1
            n_ = int( N_[c] )
                    
            SYMBAparameters = fetch.parameters_csv('SYMBA')
            #self.FundamentalValues = self.SYMBAparameters['FundamentalValues'][:, 0] #stock 0
            tau = SYMBAparameters['tau']
            kappa = SYMBAparameters['kappa']
            #print(min(tau))

            if not log_norm:
            
                x = tau.values
                x = x.astype(float)
                x -= taumin
                x /= (taumax-taumin)                                           
                #x = tau.values/taumax
                
            else:
            
                x = np.log(tau.values)
                x -= np.log(taumin)
                x /= (np.log(taumax)-np.log(taumin))

            y = kappa.values
            y -= kappamin
            y /= (kappamax-kappamin) 
            #y = kappa.values/kappamax
                         
            #y = np.log(kappa.values)
            #y -= np.log(kappamin)
            #y /= (np.log(kappamax)-np.log(kappamin))
            
            x *=resolution #resolution
            y *=resolution
            
            for s in range(self.S):

                aFtemp = fetch.saved('actionForecasting', s=f'{s}')[:, 0]
                Z[c, n_, s] = aFtemp[:, self.tInit:].mean(axis=1)
                X[c, n_, s] = x
                Y[c, n_, s] = y

        for i, ax in enumerate(axes):
 
            x = X[i].flatten()
            y = Y[i].flatten()
            z = Z[i].flatten()
            
            #breakpoint()
            #x_ = np.zeros((len(x), 2))
            #y_ = np.zeros((len(x), 1))
            #x_[:, 0] = x
            #x_[:, 1] = y
            #regr = LinearRegression()
            #res = regr.fit(x_, y_)
            #print(i, res.coef_)
        
            grid_x, grid_y = np.mgrid[res0x:resolution+1, res0y:resolution+1]
            z_interpolated = griddata((x,y), z, (grid_x, grid_y), method='linear')
            
            hue_order_position = i

            ax_temp = axes[hue_order_position]
            letter = string.ascii_lowercase[hue_order_position]

            ax_temp.text(-0.08, 1.04, letter, transform=ax_temp.transAxes, size=12, weight='bold') if hue_order_position > 0 else ax_temp.text(-0.23, 1.04, letter, transform=ax_temp.transAxes, size=12, weight='bold')
            #breakpoint()
            im = ax_temp.imshow(z_interpolated, cmap=cmap, origin='lower', vmin=0, vmax=1, aspect='auto', extent=(0, 1, 0, 1)) 

            ax_temp.set_xticks(ticks)
            ax_temp.set_xticklabels(tickLabels)
            ax_temp.set_yticks(ticks)
            ax_temp.set_yticklabels(tickLabels) if i == 0 else None

            ax_temp.set_xlabel(r'$\kappa$ (normalised)')
            ax_temp.set_ylabel(r'$\tau$ (normalised)') if i == 0 else None 


            #if i == N-1:
                #cbar = plt.colorbar(im)
                #cbar.ax.set_title(r'$\bar{a}_{\mathcal{F}}$')                

                #divider = make_axes_locatable(ax)
                #cax = divider.append_axes('right', size='50%', pad = 0.1)
                #cax.set_title(r'$\bar{a}_{\mathcal{F}}$')
                #fig.colorbar(im, cax=cax, orientation='vertical')
        
        #cax = divider.append_axes('right', size='5%', pad = 0.1)
        #cax.set_title(r'$\bar{a}_{\mathcal{F}}$')

        #fig.text(-0.01, 0.5, r'$\tau$ (normalised)', va='center', rotation='vertical')        
        #fig.text(0.5, -0.07, r'$\log{\kappa}$ (normalised)', ha='center')
        #fig.text(0.5, -0.1, r'$\kappa$ (normalised)', ha='center')
        #plt.xlabel(r'$\log{\kappa}$ (normalised)')
            
        # Adjust layout to create space between pairs of subplots
        #plt.tight_layout()

        cbaxes = fig.add_axes([0.925, 0.1, 0.03, 0.75])
        fig.colorbar(im, cax=cbaxes)
        cbaxes.set_title(r'$\bar{a}_{\mathcal{F}}$')

        fig.subplots_adjust(wspace=0.15) #, hspace=0.4)
    
        # Save the figure for kappa
        title = f'/Forecast_preference_mesh_avg_all.pdf' if not log_norm else f'/Forecast_preference_mesh_avg_all_log.pdf'
        plt.savefig(self.plotPath + title, dpi=300, bbox_inches='tight')
        plt.close()

    # Local regression (LOESS)
    def loess(self, x, y, ax, frac=0.3):
    
        # Sort the values of x
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Lowess smoothing
        lowess = sm.nonparametric.lowess
        z = lowess(y, x, frac=frac)

        # Plot the smoothed line
        ax.plot(z[:, 0], z[:, 1], color='red', linestyle='--', label='lowess')

    def success_parameters(self, exN, N):

        df = pd.DataFrame()

        Wname = r"$r_{W}$" #r"$\log_{10}{W_{\text{total}, T}} - \log_{10}{W_{\text{total}, t_{0}}}$"

        Price_tInit = self.fetch.saved('Price', s='training_0')[0, self.tInit-1]
        VolumeOwned_tInit = self.fetch.parameters_json('wallet')['InitialVolumeOwned']
        Liquidity_balance = self.fetch.parameters_json('wallet')['Liquidity balance']
        Winit = Price_tInit*VolumeOwned_tInit*(1+Liquidity_balance)
        Winit = np.log10(Winit)
        
        I = self.fetch.input_parameters()['I']
        index = range(I)

        palette = self.palette[:5]
         
        df_list = []
        
        for n in range(N):
            store = FileStorage()
            store.set_example(exN+n)
            fetch = FileFetching(store)
            
            S = fetch.input_parameters(time=True)['S']
            #S = 1
            
            for s in range(S):
                         
                W = fetch.saved('Atotal', s=str(s))[:, -1]
                W = np.log10(W)
                W -= Winit
            
                condition = fetch.input_parameters()['CognitiveTraits'][0]
                condition = biasDict[condition]
            
                df_temp = pd.DataFrame(data=W, columns=[Wname], index=index)
                df_temp['condition'] = condition
                df_temp['random_seed'] = fetch.input_parameters()['randomSeed']

                print( df_temp['condition'].unique(), np.mean(W > 0)*100, np.sum( (W > 0)*W ) )
           
                # RL parameters
                #RLparameters = fetch.parameters_csv('RL')
        
                #alphaConf = RLparameters['alphaConf']
                #alphaDisc = RLparameters['alphaDisc']
            
                #df_temp['bias'] = alphaConf-alphaDisc

                # RL parameters
                parameters = fetch.parameters_csv('SYMBA')            
                df_temp[parameters.columns[1:]] = parameters[parameters.columns[1:]]

                df_list.append(df_temp)
        
        df = pd.concat(df_list)

        df['g'] = df['g']*df['gSkewness']
        df[r'$\log{ \kappa}$'] = np.log10( df['kappa'] )
        df = df.drop(['rhoAsk', 'gSkewness', 'kappa'], axis=1)
        #biasName = r'$\Delta \alpha$'
        df = df.rename(columns={'g': r'$g$', 'tau': r'$\tau$', 'rhoBid': r'$\rho_{\text{bid}}$'}) #'bias': biasName
        
        hue_order = self.order[:5]
        
        fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
        axs = axs.flatten()
        
        y = Wname

        for columnN, (x, ax) in enumerate( zip(df.columns[3:], axs) ):
            
            legend = True if x == df.columns[-1] else False
            
            abbreviations_temp = abbreviations #[r'$B_{+}$', r'$B_{0}$', r'$B_{-}$'] if x == biasName else abbreviations
            palette_temp = palette #['r', [.3, 0, .3], 'b'] if x == biasName else palette
            
            for condition_n, condition in enumerate( abbreviations_temp ):
                dftemp = df[df.condition==condition]
                g = sns.regplot(data=dftemp, x=x, y=y, color=palette_temp[condition_n], lowess=True, scatter_kws={'alpha':.1, 's':0.05}, label=condition, ax=ax)
            
            ax.text(-0.3, 1.0, string.ascii_lowercase[columnN], transform=ax.transAxes, size=12, weight='bold')
            
            ax.axhline(y=0, color=[.5, .5, .5], linewidth=.5)
            ax.set_ylabel('') if columnN > 0 else None #if x != biasName else None

            if legend:
                                
                leg = g.legend()
                        
                for lh in leg.legend_handles:
            
                    lh.set_alpha(1)
                    lh._sizes = [50]

        
        #for x in df.columns[3:]:
        #    sns.lmplot(data=df, x=x, y=y, palette=palette, hue='condition', hue_order=abbreviations, lowess=True, scatter_kws={'alpha':.1, 's':0.1}) #order=2,

            #plt.axhline(y=0, color=[.5, .5, .5], linewidth=.5)
            
            
            #plt.savefig(self.plotPath+f'/Success_parameters_{x}', dpi=300)
            #plt.close()
  
        plt.tight_layout()

        plt.savefig(self.plotPath+'/Success_parameters.pdf', dpi=300)
        plt.close()

    def success_parameters_RL(self, exN, N):

        df = pd.DataFrame()

        Wname = r"$r_{W}$" #r"$\log_{10}{W_{\text{total}, T}} - \log_{10}{W_{\text{total}, t_{0}}}$"

        Price_tInit = self.fetch.saved('Price', s='training_0')[0, self.tInit-1]
        VolumeOwned_tInit = self.fetch.parameters_json('wallet')['InitialVolumeOwned']
        Liquidity_balance = self.fetch.parameters_json('wallet')['Liquidity balance']
        Winit = Price_tInit*VolumeOwned_tInit*(1+Liquidity_balance)
        Winit = np.log10(Winit)
        
        I = self.fetch.input_parameters()['I']
        index = range(I)

        palette = ['r',          #CB     
                  [.8, 0, .8],   #NB
                  [.3, 0, .3],   #NNB                        
                  'b',           #PB   
                  [.5, .5, .5]]  #NL
        
        cognitive_traits = ['optimistic bias', 'no bias', 'no bias individual', 'pessimistic bias', 'no learning']
        abbreviations = [r'$B_{+}$', r'$B_{0}$', r'$B_{00}$', r'$B_{-}$', r'$I_{0}$']
        biasDict = dict( zip(cognitive_traits, abbreviations) )
    
        df_list = []
        
        for n in range(N):
            store = FileStorage()
            store.set_example(exN+n)
            fetch = FileFetching(store)
            
            S = fetch.input_parameters(time=True)['S']
            #S = 3
            
            for s in range(S):
                         
                W = fetch.saved('Atotal', s=str(s))[:, -1]
                W = np.log10(W)
                W -= Winit
            
                condition = fetch.input_parameters()['CognitiveTraits'][0]
                
                if condition != 'no learning':
                
                    condition = biasDict[condition]
            
                    df_temp = pd.DataFrame(data=W, columns=[Wname], index=index)
                    df_temp['condition'] = condition
                    df_temp['random_seed'] = fetch.input_parameters()['randomSeed']

                    print( df_temp['condition'].unique(), np.mean(W > 0)*100, np.sum( (W > 0)*W ) )
           
                    # RL parameters
                    RLparameters = fetch.parameters_csv('RL')
        
                    alphaConf = RLparameters['alphaConf']
                    alphaDisc = RLparameters['alphaDisc']
            
                    df_temp[r'$\alpha_{+}$'] = alphaConf
                    df_temp[r'$\alpha_{-}$'] = alphaDisc                
                    df_temp['bias'] = alphaConf-alphaDisc

                    # RL parameters
                    #parameters = fetch.parameters_csv('SYMBA')            
                    #df_temp[parameters.columns[1:]] = parameters[parameters.columns[1:]]

                    df_list.append(df_temp)
        
        df = pd.concat(df_list)

        biasName = r'$\Delta \alpha$'
        df = df.rename(columns={'bias': biasName})
        
        hue_order = self.order[:5]
        
        fig, axs = plt.subplots(1, 3, figsize=(6, 3), sharey=True)
        axs = axs.flatten()
        
        y = Wname

        for columnN, (x, ax) in enumerate( zip(df.columns[3:], axs) ):
            
            legend = True if x == df.columns[-1] else False
            
            abbreviations_temp = [r'$B_{+}$', r'$B_{0}$', r'$B_{-}$'] if x == biasName else abbreviations
            palette_temp = ['r', [.3, 0, .3], 'b'] if x == biasName else palette
            
            for condition_n, condition in enumerate( abbreviations_temp ):
                dftemp = df[df.condition==condition]
                g = sns.regplot(data=dftemp, x=x, y=y, color=palette_temp[condition_n], lowess=True, scatter_kws={'alpha':.1, 's':0.05}, label=condition, ax=ax)
            
            ax.text(-0.3, 1.0, string.ascii_lowercase[columnN], transform=ax.transAxes, size=12, weight='bold')
            
            ax.axhline(y=0, color=[.5, .5, .5], linewidth=.5)
            ax.set_ylabel('') if x != biasName else None

            if legend:
                                
                leg = g.legend()
                        
                for lh in leg.legend_handles:
            
                    lh.set_alpha(1)
                    lh._sizes = [50]

        
        #for x in df.columns[3:]:
        #    sns.lmplot(data=df, x=x, y=y, palette=palette, hue='condition', hue_order=abbreviations, lowess=True, scatter_kws={'alpha':.1, 's':0.1}) #order=2,

            #plt.axhline(y=0, color=[.5, .5, .5], linewidth=.5)
            
            
            #plt.savefig(self.plotPath+f'/Success_parameters_{x}', dpi=300)
            #plt.close()
  
        plt.tight_layout()

        plt.savefig(self.plotPath+'/Success_parameters_RL.pdf', dpi=300)
        plt.close()
    
    def LSE_stock(self):

        LSE = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)
        LSEstock = self.fetch.input_parameters()['LSEstock']
        Price = LSE[LSEstock].round(2)[:self.tInit] 
        
        plt.figure(figsize=(8, 4))
        plt.plot(Price, color='k')
        plt.ylabel('Price')
        plt.xlabel('Trading days (t)')

        plt.savefig(self.plotPath+f'/LSE_stock_{LSEstock}.pdf', dpi=300)
        plt.close()

    def LSE_data(self):

        ### Empirical LSE data ###
        try:
            LSE = pd.read_csv('LSE_data/LSE_data_closing_prices.csv', index_col=False)
        except:
            LSE = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)
        #breakpoint()
        
        stockSelection = (LSE.diff(periods=1, axis=0).iloc[1:] == 0).mean() < .5 #active at least more than 50% of times
        stockSelection = pd.DataFrame(stockSelection, columns=['Active'])
        stockSelection = stockSelection[stockSelection['Active']==True].index

        self.LSE = LSE[stockSelection]
        
        self.LSEr = np.log(self.LSE[stockSelection]).diff(periods=1, axis=0).iloc[1:]

    def AC_volatility(self): #TO BE FIXED!

        Tw = 5 #trading week-days
        Tm = 21 #monthly trading days
        Ty = 252 #yearly trading days
        
        hue = 'Data'
        column = 'AC'

        fig, ax = plt.subplots(figsize=(6, 6))
        
        sim = pd.DataFrame( data = self.rdf.rolling(window=Tw).var().iloc[Tw:].apply(lambda x : sm.tsa.acf(x, nlags=1)[-1]), columns=[column])
        LSE = pd.DataFrame( data = self.LSEr.rolling(window=Tw).var().iloc[Tw:].apply(lambda x : sm.tsa.acf(x, nlags=1)[-1]), columns=[column] )
        #Wiener = pd.DataFrame( data = brownian.rolling(window=Tw).var().apply(lambda x : sm.tsa.acf(x, nlags=1)[-1]), columns=[column] )
            
        sim_hue = f'pSYMBA'
        LSE_hue = f'LSE'
        #Wiener_hue = f'Wiener ($\sigma^2={str(deltaBrownian)}$)'
            
        sim[hue] = sim_hue
        LSE[hue] = LSE_hue
        #Wiener[hue] = Wiener_hue
            
        data_hue_list = [sim_hue, LSE_hue] #, Wiener_hue]
        ax.axvline(x=sm.tsa.acf(self.LSEr[self.LSEstock].rolling(window=Tw).var(), nlags=1)[-1], color=[0, 0, .5], label=self.LSEstock)
        #ax.axvline(x=0, color=[.8, .8, .8], linestyle=':', linewidth=1.5)
            
        #data = pd.concat([sim, LSE, Wiener], ignore_index=True)
        data = pd.concat([sim, LSE], ignore_index=True)
        
        for hue_n, data_hue in enumerate(data_hue_list):
            sns.kdeplot(data=data[data[hue]==data_hue], x=column, common_norm=False, color=colors[hue_n], fill=False, bw_adjust=2, ax=ax, label=data_hue)
            
        ax.legend()
        #ax.set_xlim([-.45, .45])
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.savefig(self.plotPath+'/AC_volatility.pdf', dpi=100)
        plt.close()


############## 

    def cum_distr_abs_ret(self, ex, N, units=False):

        errorbar = 'se' #'sd'

        x_name = '|r|'
        y_name = 'Cumulative frequency'

        cognitive_traits = ['optimistic bias', 'no bias', 'no bias individual', 'pessimistic bias', 'no learning']
        abbreviations = [r'$B_{+}$', r'$B_{0}$', r'$B_{00}$', r'$B_{-}$', r'$I_{0}$']
        biasDict = dict( zip(cognitive_traits, abbreviations) )
        
        df_list = []

        X = np.arange(0, .6, 0.0005)

        self.LSE_data() #fetch LSE data and get returns

        df_list = []

        #N = 200
        for n in self.LSEr.columns: #[:N]:
            
            ##SortedAbsr = sorted(self.LSEr[n].abs().values)
            Absr = self.LSEr[n].abs().values
            y = np.array([(Absr >= x).sum()/len(self.LSEr) for x in X])
            #ax.step(X, y, color=[.5, .5, .5], alpha=.1, label='LSE')

            df_temp = pd.DataFrame(data=y, columns=[y_name]) #, index=index)
            df_temp[x_name] = X
            df_temp['condition'] = 'LSE'
            df_temp['rs'] = n
            
            df_list.append(df_temp)

        df_LSE = pd.concat(df_list)

        if units:
            ax = sns.lineplot(data=df_LSE, x=x_name, y=y_name, hue='condition', palette=['k'], units='rs', estimator=None, lw=0.1) #,

        #pSYMBA

        for n in range(N):
             
            store = FileStorage()
            store.set_example(ex + n)  # Change example every subplot
            fetch = FileFetching(store)
            random_seed = fetch.input_parameters()['randomSeed']

            fundamentalValueTrue = self.fetch.saved('fundamentalValueTrue', s='0', verbose=False)[0, self.tInit:]
            returns = np.log(fundamentalValueTrue[1:])-np.log(fundamentalValueTrue[:-1])
            Absr = abs(returns)
            y = np.array([(Absr >= x).sum()/len(returns) for x in X])            
            df_temp = pd.DataFrame(data=y, columns=[y_name]) #, index=index)
            df_temp[x_name] = X
            df_temp['condition'] = '$f$'
            df_temp['random_seed'] = random_seed
            #df_temp['rs'] = df_temp['random_seed']+df_temp['sim'] #'-'+df_temp['sim']

            df_list.append(df_temp)

            condition = fetch.input_parameters()['CognitiveTraits'][0]
                    
            for s in range(self.S):

                Price = fetch.saved('Price', s=f'{s}')[0, self.tInit:]
                returns = np.log(Price[1:])-np.log(Price[:-1])
                Absr = abs(returns)
                y = np.array([(Absr >= x).sum()/len(returns) for x in X])
                #ax.step(X, y, color=[.5, 0, 0], alpha=.4, label='pSYMBA')

                df_temp = pd.DataFrame(data=y, columns=[y_name]) #, index=index)
                df_temp[x_name] = X
                df_temp['condition'] = biasDict[condition]
                df_temp['random_seed'] = random_seed
                df_temp['sim'] = s
                df_temp['rs'] = df_temp['random_seed']+df_temp['sim'] #'-'+df_temp['sim']
                
                df_list.append(df_temp)

        df = pd.concat(df_list)

        palette = self.palette[:6]

        hue_order = self.order[:6] #, 'LSE']
        
        if not units:
            ax = sns.lineplot(data=df, x=x_name, y=y_name, hue='condition', hue_order=hue_order, palette=palette, alpha=.7, errorbar=errorbar) #, style='rs')
        else:                    
            sns.lineplot(data=df, x=x_name, y=y_name, hue='condition', hue_order=hue_order, palette=palette, alpha=.7, lw=0.1, units='rs', estimator=None, ax=ax)

        if not units:
            sns.lineplot(data=df_LSE, x=x_name, y=y_name, hue='condition', palette=['k'], ax=ax, errorbar=errorbar) #,        
 
        #Normal    
        #Absr = abs(np.random.normal(scale=0.02, size=252*10))
        #y = np.array([(Absr >= x).sum()/len(Absr) for x in X])
        #ax.step(X, y, color=[0, 0, 0], alpha=.8, label='Normal '+r'$\sigma^2=0.02$')

        ##LSE stock of reference
        ##SortedAbsr = sorted(self.LSEr[self.LSEstock].abs().values)
        #Absr = self.LSEr[self.LSEstock].abs().values
        #y = np.array([(Absr >= x).sum()/len(self.LSEr) for x in X])
        #ax.step(X, y, color=[0, 0, .6], alpha=1, label=self.LSEstock)

        #ax.set_yscale('log')
                
        #def legend_without_duplicate_labels(ax):
        #    handles, labels = ax.get_legend_handles_labels()
        #    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        #    leg = ax.legend(*zip(*unique))
        #    for lh in leg.legendHandles: 
        #        lh.set_alpha(1)

        #legend_without_duplicate_labels(ax)

        ##plt.xlabel(r'$|r|$')
        ##plt.ylabel('Cumulative frequency')
        ##sns.histplot(data=SF2, x=column, hue=hue, alpha=.35, palette=['r', 'b'])
        plt.yscale('log')
        plt.savefig(self.plotPath+f'/SF5_units_{units}_errorbar_{errorbar}.pdf', dpi=100)
        plt.close()


def get_parser():
    
    #https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', nargs='?', const=True, type=int, help = 'please give the comparison number to analyse', required=False, default=-1)
    #parser.add_argument('--datasetEnd', nargs='?', const=True, type=int, help = 'please give the name of a dataset folder', required=False, default='all')
    
    args = parser.parse_args()
    
    return args
     
if __name__ == '__main__':

    args = get_parser()
    n = args.n
    plot = Plot(n)
    
    ex = plot.exStart
    N = plot.N
    
    #plot.market_dynamics()
    [plot.price_dynamics(ex+40, s=s) for s in range(plot.S)]

    #plot.scatter_plot_alphas(ex)
    #plot.cum_distr_abs_ret(ex, N, units=False)
    
    #log_norm_tau = True
    #plot.forecast_preference(ex, N=50)
    #plot.forecast_preference_avg_all(ex, N, log_norm_tau)    
    #plot.forecast_preference_avg_all(ex, N, log_norm=False)

    #plot.phase_plot(ex)
    #plot.kirman_plot(ex, N)
    #plot.kirman_plot_grid(ex, N)
    #plot.success_parameters(ex, 5)
    #plot.success_parameters_RL(ex, 5)    
    #plot.LSE_stock()
    
    #plot.wealth_change(ex, N) #moved to development
    #plot.wealth_change_fromtraining(ex, N) #moved to development
