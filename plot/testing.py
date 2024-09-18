#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:46:49 2022

@author: svrizzi
"""

import sys
from tabnanny import verbose
from matplotlib.lines import lineStyles

from pyparsing import line
sys.path.append('..') if '..' not in sys.path else None  #go to parent folder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
import pandas as pd
import scipy.stats
from scipy.stats import norm
import os
import imageio.v2 as imageio

import matplotlib as mpl
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import cycle, islice
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
from sklearn.manifold import TSNE

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial']})
import string

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import matplotlib.colors as mcolors

sliding_window_view = np.lib.stride_tricks.sliding_window_view

class PlotTesting:

    def __init__(self, fetch):
        
        self.fetch = fetch
    
        self.input_parameters()
        self.trader_parameters()        
        self.market_variables()
        self.LSE_data()
        self.create_plot_folders()

    def create_plot_folders(self):

        self.example_folder_path = self.fetch.example_folder_path

        #Save plots in folders
        self.plotFolderPath = self.fetch.plotFolderPath

        #Create folders
        try:
            os.mkdir(self.plotFolderPath) if not os.path.isdir(self.plotFolderPath) else None #create Data folder for all data files, in case it does not exist
        except:
            self.plotFolderPath += '../'
            os.mkdir(self.plotFolderPath) if not os.path.isdir(self.plotFolderPath) else None #create Data folder for all data files, in case it does not exist

        self.plotPath = self.plotFolderPath+self.example_folder_path()
        os.mkdir(self.plotPath) if not os.path.isdir(self.plotPath) else None #create Data folder for all data files, in case it does not exist

        self.plotPath = self.plotFolderPath+self.example_folder_path()+'/testing'
        os.mkdir(self.plotPath) if not os.path.isdir(self.plotPath) else None #create Data folder for all data files, in case it does not exist

        self.plotPathStylisedFacts = self.plotPath+'/Stylised_facts'
        os.mkdir(self.plotPathStylisedFacts) if not os.path.isdir(self.plotPathStylisedFacts) else None #create Data folder for all data files, in case it does not exist
        
    def input_parameters(self):

        #Input parameters
        self.InputParameters = self.fetch.input_parameters()
        self.I = self.InputParameters['I']
        self.J = self.InputParameters['J']

        self.saveMicro = self.InputParameters['saveMicro']
        
        self.InputTimeParameters = self.fetch.input_parameters(time=True)
        self.T = self.InputTimeParameters['T']
        self.S = self.InputTimeParameters['S']

    def trader_parameters(self):

        #RL parameters
        self.RLparameters = self.fetch.parameters_csv('RL')
        self.CognitiveTraits = self.InputParameters['CognitiveTraits']
        
        self.beta = self.RLparameters['beta']
        self.alphaConf = self.RLparameters['alphaConf']
        
        if 'optimistic bias' in self.CognitiveTraits:
            self.alphaDisc = self.RLparameters['alphaDisc']
        
        #plt.figure(figsize=(6, 4.5))
        #plt.scatter(self.alphaConf[1:], self.alphaDisc[1:], color='k', label='agent 2 to $I$')
        #plt.scatter(self.alphaConf[0], self.alphaDisc[0], color='r', label='agent 1')
        #plt.xlabel(r'$\alpha_{+}$')
        #plt.ylabel(r'$\alpha_{-}$')
        #plt.legend(bbox_to_anchor=(1.05, .55))
        #plt.tight_layout()
        #plt.show()
        
        #breakpoint()
        
        #SYMBA parameters
        self.tInit = self.fetch.parameters_json('SYMBA')['tInit']

        self.SYMBAparameters = self.fetch.parameters_csv('SYMBA')
        #self.FundamentalValues = self.SYMBAparameters['FundamentalValues'][:, 0] #stock 0
        self.rhoAsk = self.SYMBAparameters['rhoAsk']
        self.rhoBid = self.SYMBAparameters['rhoBid']
        self.tau = self.SYMBAparameters['tau']
        self.g = self.SYMBAparameters['g']
        self.kappa = self.SYMBAparameters['kappa']
        #self.ClosingLimit = self.SYMBAparameters['ClosingLimit']

    def market_variables(self):

        self.LSEstock = self.InputParameters['LSEstock']

        #Market variables
        self.Price = np.zeros((self.S, self.T-self.tInit))
        self.Volume = np.zeros((self.S, self.T-self.tInit))
        self.BidAskSpread = np.zeros((self.S, self.T-self.tInit))
        self.fundamentalValueTrue = np.zeros((self.S, self.T-self.tInit))

        if self.saveMicro:
            self.VolumeOwned = np.zeros((self.S, self.I, self.T-self.tInit))
            self.Aequity = np.zeros((self.S, self.I, self.T-self.tInit))
            self.Wallet = np.zeros((self.S, self.I, self.T-self.tInit))
            self.AvailableMargin = np.zeros((self.S, self.I, self.T-self.tInit))
            self.Atotal = np.zeros((self.S, self.I, self.T-self.tInit))
    
        for s in range(self.S):
        
            sFolder = str(s)
            
            self.Price[s] = self.fetch.saved('Price', s=sFolder, verbose=True)[0, self.tInit:]
            self.Volume[s] = self.fetch.saved('Volume', s=sFolder, verbose=False)[0, self.tInit:]
            self.BidAskSpread[s] = self.fetch.saved('BidAskSpread', s=sFolder, verbose=False)[0, self.tInit:]
            self.fundamentalValueTrue[s] = self.fetch.saved('fundamentalValueTrue', s=sFolder, verbose=False)[0, self.tInit:]
            
            if self.saveMicro:
                
                self.VolumeOwned[s] = self.fetch.saved('VolumeOwned', s=sFolder, verbose=False)[:, 0, self.tInit:]
                self.Aequity[s] = self.fetch.saved('Aequity', s=sFolder, verbose=False)[:, 0, self.tInit:]
                self.Wallet[s] = self.fetch.saved('Wallet', s=sFolder, verbose=False)[:, self.tInit:]
                self.AvailableMargin[s] = self.fetch.saved('AvailableMargin', s=sFolder, verbose=False)[:, self.tInit:]
                self.Atotal[s] = self.fetch.saved('Atotal', s=sFolder, verbose=False)[:, self.tInit:]

        self.r = np.log(self.Price[:, 1:]) - np.log(self.Price[:, :-1]) #returns
        self.rdf = pd.DataFrame(data=self.r.T, columns=list(range(self.S)))

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

    def RL_variable(self, variableName, time=None, s=0):
       
       NactionsForecasting = 2 #3
       
       T = self.T if not time else 1
       
       if variableName == 'QForecasting':
          variable = np.zeros((self.I, T, 2, NactionsForecasting))
       elif variableName == 'QTrading':          
          variable = np.zeros((self.I, T, NactionsForecasting, 8, 3))
       elif variableName == 'QPricing':
          variable = np.zeros((self.I, T, 2, 2, 5))
       elif variableName == 'actionForecasting':
          variable = np.zeros((self.I, T)) 
       elif variableName == 'actionTrading':
          variable = np.zeros((self.I, T))    
       elif variableName == 'stateTrading':
          variable = np.zeros((self.I, T, 2))
            
       folderName = f'{s}'
            
       tSkip = self.tInit
       #tSkip += max(self.tau)*(s<1)            
                                
       if variableName in ['actionForecasting', 'actionTrading', 'stateTrading', 'stateForecasting']:
          variable = self.fetch.saved(variableName, folderName)[:, 0, tSkip:]
       else:
          variable = self.fetch.saved(variableName, folderName)[:, tSkip:]

       return variable

    ################################
    ############# Plots ############
    ################################

    def mean_square_displacement(self):
    
        import matplotlib
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = axs.ravel()
        
        lags = np.logspace(4, 30, base=1.2).astype(int) #int(self.T/2))
        lags = list( set(lags) )
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.S)
                
        SquaredPriceResiduals = np.zeros((self.S, len(lags)))
        SquaredPriceResidualsLSE = np.zeros((len(lags)))
        slopes = np.zeros(self.S)
        
        from scipy.stats import linregress
        
        for s in range(self.S):

            for tn, t_ in enumerate(lags):
                
                PriceWindows = sliding_window_view(self.Price[s], window_shape = t_)
                price_residuals = lambda x: (np.log(x[-1]) - np.log(x[0]))**2
                
                SquaredPriceResiduals[s, tn] = np.array([price_residuals(x) for x in PriceWindows]).mean()
                
                axs[0].scatter(t_, SquaredPriceResiduals[s, tn], color=cm.jet(norm(s)), alpha=.5, s=1)

                PriceWindows = sliding_window_view(self.LSE[self.LSEstock].values, window_shape = t_)
                price_residuals = lambda x: (np.log(x[-1]) - np.log(x[0]))**2
                SquaredPriceResidualsLSE[tn] = np.array([price_residuals(x) for x in PriceWindows]).mean()

                axs[0].scatter(t_, SquaredPriceResidualsLSE[tn], color='k', alpha=.8, s=2, marker='+')
                    
            result = linregress(np.log( lags ), np.log( SquaredPriceResiduals[s] ))
            slopes[s] = result.slope
        
        #plt.plot(plt.xlim(), plt.ylim(), linewidth=.5, color='k', alpha=.4) #identity line as reference for diffusion
        axs[0].set_xlabel('Time lag '+r'$(t-\tau)$')
        axs[0].set_ylabel('Mean '+r'$[r(t-\tau, t)^2]$')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        
        #pSYMBA
        df = pd.DataFrame(data=slopes, columns=['slope'])
        df['data'] = 'pSYMBA'
        
        #LSE
        result = linregress(np.log( lags ), np.log( SquaredPriceResidualsLSE ) )
        #df = df.append({"slope": result.slope, "data": "LSE"}, ignore_index=True)
        
        sns.stripplot(data=df, x='data', y='slope', palette=['k', 'r'], ax=axs[1])             
        axs[1].axhline(y=result.slope, color=[.5, 0, 0], linestyle='-', label='LSE')
        axs[1].axhline(y=1, color='k', alpha=.7, linestyle='-')
        axs[1].set_xlabel('')
        plt.legend()
        plt.savefig(self.plotPath+f'/mean_square_displacement.pdf', dpi=120)  
    
    def QF_convergence_learning(self):

        QF = self.RL_variable('QForecasting')

        T = range(self.T-self.tInit)
        traders = np.random.uniform(self.I, size=1000).astype(int)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
        axs = axs.flatten()

        for axn, ax in enumerate( axs ):

            [ ax.plot(T, np.squeeze( QF[trader, :, axn, 0] ), color='r', alpha=1/self.tau[trader], linewidth=10/self.tau[trader]) for trader in traders ]
            [ ax.plot(T, np.squeeze( QF[trader, :, axn, 1] ), color='b', alpha=1/self.tau[trader], linewidth=10/self.tau[trader]) for trader in traders ]
            #[ plt.plot(T, np.squeeze( QF[trader, :, 2] ), color='k', alpha=.02) for trader in traders ]
            ax.set_xlim([0, self.T])
            ax.set_ylim([QF.min(), QF.max()*1.05])
            #plt.ylim([-10, QF.max()*1.05])
            ax.set_xlabel(r'$t$')
            ax.set_ylabel(r'$Q_F$')
            
        plt.legend()
        plt.savefig(self.plotPath+'/convergence_QF_micro.pdf', dpi=120)

        plt.close()

    # Price dynamics
    def price_dynamics_fundamental_value(self):
                        
        for s in range(self.S):

            # Assuming kappa is a numpy array of size (num_agents,)
            # Normalize the kappa values between 0 and 1
            norm = plt.Normalize(vmin=np.min(self.kappa), vmax=np.max(self.kappa))

            # Map each agent's kappa to a color using the rainbow colormap
            colors = cm.rainbow(norm(self.kappa))
            
            Price = self.fetch.saved('Price', s=f'{s}')[0, self.tInit:]
            fundamentalValue = self.fetch.saved('fundamentalValue', s=f'{s}')[:, 0]
 
            plt.figure()

            # Loop over agents and plot each one with the corresponding color
            for i in range(fundamentalValue.shape[0]):  # assuming shape[0] is the number of agents
                plt.plot(fundamentalValue[i], color=colors[i], alpha=0.05)
                
            plt.plot(self.Price[s, -self.tInit:], color='k')
            
            #sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
            #sm.set_array([])
            #plt.colorbar(sm, label="$\kappa$")

            title = f'/FV_dynamics_stock{s}_testing.png'
        
            plt.savefig(self.plotPath+title, dpi=300)
        
    # Price dynamics
    def price_dynamics(self):
                        
        for s in range(self.S):
        
            actionForecasting = self.RL_variable('actionForecasting', s=s)
        
            fig, ax = plt.subplots(figsize=(12, 2.5))

            #Price dynamics
            #cmap = mcolors.LinearSegmentedColormap.from_list("", ["darkred", "lightgrey", "darkblue"])
            cmap = plt.cm.get_cmap('coolwarm_r')
            norm = mcolors.Normalize(vmin=0, vmax=1)
            aFmean = actionForecasting.mean(axis=0)
        
            for t in range( len(self.Price[s, :])-1 ):
                #plt.plot(self.Price[0, :], color=cmap(actionForecasting.mean(axis=0)), label='pSYMBA')
                ax.plot([t, t+1], [self.Price[s, t], self.Price[s, t+1]], color=cmap(norm(aFmean[t])))
        
            #cmap = 'coolwarm_r'
            #plt.scatter(range(len(self.Price[0, :])), self.Price[0, :], c=actionForecasting.mean(axis=0), cmap='coolwarm_r', marker='o', edgecolor='none', vmin=0, vmax=1)
            #plt.colorbar(label='Fundamentalism')
        
            ax.plot(self.fundamentalValueTrue[s, :], color='k', alpha=0.3, label=r'$f$')

            # Add color bar showing information values
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, label='Fundamentalism', shrink=0.8, pad=0.02)

            # Adjust layout to accommodate the colorbar
            plt.subplots_adjust(right=0.85)  # Increase right margin to make space for colorbar
            cbar.ax.yaxis.set_ticks_position('right')  # Position colorbar ticks on the left side
        
            plt.ylim([0, None])
            plt.xlabel('Time (days)')
            plt.ylabel('Stock price (€)')
            plt.legend()
        
            plt.tight_layout()

            title = f'/Price_dynamics_stock{s}_testing.png'
        
            plt.savefig(self.plotPath+title, dpi=300)
  
    # Price dynamics
    def market_dynamics(self):
    
        low = np.argmin([min(self.Price[s]) for s in range(self.S)])
        high = np.argmax([max(self.Price[s]) for s in range(self.S)])
        
        sample = list(range(self.S))
        sample.remove(low) if self.S > 1 else None
        sample.remove(high) if high != low else None
        sample = [low, high, np.random.choice(sample)]
        
        plt.figure(figsize=(6, 7))

        ax1 = plt.subplot(411)
        [ax1.plot(self.Price[s], color='k', alpha=.2) for s in sample] #, label='pSYMBA simulation'
        ax1.plot(self.Price.mean(axis=0), color='r', label=f'grand mean pSYMBA (S={self.S})')
        ax1.set_ylabel('Stock price (€)')
        plt.tick_params('x', labelbottom=False)
        plt.legend()
        
        ax2 = plt.subplot(412, sharex=ax1)
        [ax2.plot(self.Volume[s], color='k', alpha=.2) for s in sample]
        ax2.plot(self.Volume.mean(axis=0), color='r')
        ax2.set_ylabel('Trading volume')
        plt.tick_params('x', labelbottom=False)
    
        ax3 = plt.subplot(413, sharex=ax1)
        [ax3.plot(self.BidAskSpread[s], color='k', alpha=.2) for s in sample]
        ax3.plot(self.BidAskSpread.mean(axis=0), color='r')
        ax3.set_ylabel('Bid-Ask spread (€)')
        ax3.set_yscale('log')
        plt.tick_params('x', labelbottom=False)
    
        ax4 = plt.subplot(414, sharex=ax1)

        volatility = [sliding_window_view((self.r[s]), window_shape = min(5, int(self.T/2))).var(axis=1) for s in sample]
        for s, sim in enumerate(sample):
            ax4.plot(volatility[s], color='k', alpha=.2)
        ax4.plot(np.array(volatility).mean(axis=0), color='r')
        ax4.set_xlabel('Time step (trading days)')
        ax4.set_ylabel('Volatility ( Var[r] )')
        ax4.set_yscale('log')
        #ax4.yaxis.major.formatter.set_powerlimits((-6,-2))
	
        plt.tight_layout()

        plt.savefig(self.plotPath+f'/Market_dynamics_stock0.pdf', dpi=120)

    def returns_distributions(self):

        plt.figure(figsize=(8, 8))

        #for s in range(self.S):

            #ax = plt.subplot(5, 4, s+1)

            #sns.histplot(data=self.rdf, x=s, binrange=[-.5, .5], bins=20, stat='density', common_norm=False, alpha=.35, ax=ax)
            #plt.tick_params('x', labelbottom=False) if s < 16 else None
            #ax.set_xlabel('')
            #ax.set_xlabel(str(s))

        self.rdf.plot.hist(bins=50, subplots=True, layout=(4, 5), title=range(self.S), legend=False, sharex=True, sharey=True, figsize=(12, 12), logy=True)

        plt.tight_layout()
        plt.savefig(self.plotPath+'/r_dist', dpi=100)
        plt.close()

        ###

        column = '$r$' #r'$\log(r)$'
        sim = pd.DataFrame( data = self.rdf.stack(), columns=[column])
        LSE = pd.DataFrame( data = self.LSEr.stack(), columns=[column] )
        LSEstock = pd.DataFrame( data = self.LSEr[self.LSEstock].values, columns=[column] )
        
        #breakpoint()
        
        hue = 'Data'
        sim[hue] = 'pSYMBA'
        LSE[hue] = 'LSE'
        LSEstock[hue] = self.LSEstock
                
        colors = [[.5, 0, 0], [.85, .85, .85], [0, 0, .5]]
        
        logr = pd.concat([sim, LSE, LSEstock], ignore_index=True)

        fig, ax = plt.subplots(figsize=(9, 6))        
        #ax = sns.histplot(data=logr, x=column, hue=hue, binrange=[-.5, .5], bins=100, stat='density', common_norm=False, kde=True, kde_kws={'bw_adjust':30}, alpha=.35, palette=['r', 'b', 'g'])
        sns.histplot(data=logr, x=column, hue=hue, binrange=[-.5, .5], bins=100, stat='density', common_norm=False, alpha=.5, palette=colors, ax=ax)

        #breakpoint()

        sns.kdeplot(data=logr[logr[hue] == 'pSYMBA'], x=column, color=colors[0], bw_adjust=7.5, ax=ax, label='pSYMBA')
        sns.kdeplot(data=logr[logr[hue] == 'LSE'], x=column, color=[.5, .5, .5], bw_adjust=20, ax=ax, label='LSE')
        sns.kdeplot(data=logr[logr[hue] == self.LSEstock], x=column, color=colors[2], bw_adjust=7.5, ax=ax, label=self.LSEstock)
                
        x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
        x_pdf = np.linspace(x0, x1, 1000)
        y_pdf = scipy.stats.norm.pdf(x_pdf, scale=0.02)

        ax.plot(x_pdf, y_pdf, color='k', linestyle=':', lw=1.5, alpha=1, label='Normal distribution ($\sigma^2$=0.02)')
        ax.legend()
        
        ax.set_yscale('log')
        ax.set_xlim([-0.6,0.6])
        ax.set_ylim([10**-(4.5),10**2])
        #ax = sns.displot(data=logr, x=column, hue=hue, kind="kde", bw_adjust=.25, palette=['r', 'b'], facet_kws={'xlim': [-.5, .5]})
        plt.savefig(self.plotPath+'/Log_r_SYMBAfig4.pdf', dpi=100)
        plt.close()

    def SF_returns(self):
        #argsort = np.argsort(self.rdf.mean(axis=0))
        #self.rdf.mean(axis=0).plot()
        #plt.errorbar(range(self.S), self.rdf.mean(axis=0).loc[argsort], yerr=self.rdf.std(axis=0).loc[argsort], color='k', label='pSYMBA')
        #sns.violinplot(data=rdf, x="class", y="age")
        #plt.errorbar(, ..., yerr=..., color='b', label='LSE')

        alphaLSE=.15

        fig = plt.figure(figsize=(9, 6.6))

        ax1 = plt.subplot(121)

        ax1.axvline(x=0, color='k', alpha=.3)
        ax1.axhline(y=0, color='k', alpha=.3)

        ax1.scatter(self.LSEr.mean(axis=0), self.LSEr.var(axis=0), color=[.4, .4, .4], alpha=alphaLSE) #, label='LSE (stock selection)')
        ax1.scatter(self.LSEr[self.LSEstock].mean(), self.LSEr[self.LSEstock].var(), color=[0, 0, .5], alpha=1) #, label=self.LSEstock)
        ax1.scatter(self.rdf.mean(axis=0), self.rdf.var(axis=0), color=[.5, 0, 0], alpha=.25) #, label='pSYMBA simulation')
        
        ax1.scatter(self.LSEr.mean(axis=0).mean(), self.LSEr.var(axis=0).mean(), marker='x', color='k') #, label='LSE (grand mean)')
        ax1.scatter(self.rdf.mean(axis=0).mean(), self.rdf.var(axis=0).mean(), color=[.5, 0, 0], marker='x') #, label=f'pSYMBA grand mean (S={self.S})')

        ax1.set_xlim([-0.002, 0.002])
        ax1.set_ylim([-10**(-4.5), 0.004])

        #ax1.xaxis.major.formatter.set_powerlimits((-6,-3))        
        #ax1.yaxis.major.formatter.set_powerlimits((-6,-3))

        ax1.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

        ax1.set_xlabel(r'$\mu$')
        ax1.set_ylabel(r'$\sigma^2$')
        #ax1.set_xscale('log')
        #ax1.set_yscale('log')
        #ax1.legend()
        #ax1.title('Returns (log €)')


        ax2 = plt.subplot(122)

        ax2.axvline(x=0, color='k', alpha=.3)
        ax2.axhline(y=0, color='k', alpha=.3)

        ax2.scatter(self.LSEr.skew(axis=0), self.LSEr.kurt(axis=0), color=[.55, .55, .55], alpha=alphaLSE, label='LSE (stock selection)')
        ax2.scatter(self.rdf.skew(axis=0), self.rdf.kurt(axis=0), color=[.5, 0, 0], alpha=.4, label='pSYMBA simulation')

        ax2.scatter(self.LSEr.skew(axis=0).mean(), self.LSEr.kurt(axis=0).mean(), marker='x', color='k', label='LSE (grand mean)')
        ax2.scatter(self.rdf.skew(axis=0).mean(), self.rdf.kurt(axis=0).mean(), color=[.5, 0, 0], marker='x', label=f'pSYMBA grand mean (S={self.S})')
        ax2.scatter(self.LSEr[self.LSEstock].skew(), self.LSEr[self.LSEstock].kurt(), color=[0, 0, .5], alpha=1, label=self.LSEstock)

        #ax2.set_xlim([-10, 10])
        #ax2.set_ylim([-5, 250])

        #Lower bound K = Sk^2 + 1
        x = np.arange(ax2.get_xlim()[0], ax2.get_xlim()[1], 0.01)
        y = (x**2+1)-3
        ax2.plot(x, y, color='k', alpha=.1, linestyle='--', label='Kurtosis lower bound')

        ax2.set_xscale('symlog')
        ax2.set_yscale('symlog')

        ax2.set_xlabel('Skew[r]='+r'$\frac{{{\mu_3}}}{{{\sigma^3}}}$')
        ax2.set_ylabel('(Excess) Kurt[r]='+r'$\frac{{{\mu_4}}}{{{\sigma^4}}} - 3$')
        #ax2.title('Returns')
        #plt.legend(bbox_to_anchor=(1.1, .55))
        plt.tight_layout()
        fig.subplots_adjust(top=0.80)
        fig.legend( ncol=3, bbox_to_anchor=(0.87, 0.97) )
        
        plt.savefig(self.plotPathStylisedFacts+f'/SF_2Dplots.pdf', dpi=100)

        plt.close()

    def returns_distributions(self):

        plt.figure(figsize=(8, 8))

        #for s in range(self.S):

            #ax = plt.subplot(5, 4, s+1)

            #sns.histplot(data=self.rdf, x=s, binrange=[-.5, .5], bins=20, stat='density', common_norm=False, alpha=.35, ax=ax)
            #plt.tick_params('x', labelbottom=False) if s < 16 else None
            #ax.set_xlabel('')
            #ax.set_xlabel(str(s))

        self.rdf.plot.hist(bins=50, subplots=True, layout=(4, 5), title=range(self.S), legend=False, sharex=True, sharey=True, figsize=(12, 12), logy=True)

        plt.tight_layout()
        plt.savefig(self.plotPath+'/r_dist', dpi=100)
        plt.close()

        ###

        column = '$r$' #r'$\log(r)$'
        sim = pd.DataFrame( data = self.rdf.stack(), columns=[column])
        LSE = pd.DataFrame( data = self.LSEr.stack(), columns=[column] )
        LSEstock = pd.DataFrame( data = self.LSEr[self.LSEstock].values, columns=[column] )
        
        #breakpoint()
        
        hue = 'Data'
        sim[hue] = 'pSYMBA'
        LSE[hue] = 'LSE'
        LSEstock[hue] = self.LSEstock
                
        colors = [[.5, 0, 0], [.85, .85, .85], [0, 0, .5]]
        
        logr = pd.concat([sim, LSE, LSEstock], ignore_index=True)

        fig, ax = plt.subplots(figsize=(9, 6))        
        #ax = sns.histplot(data=logr, x=column, hue=hue, binrange=[-.5, .5], bins=100, stat='density', common_norm=False, kde=True, kde_kws={'bw_adjust':30}, alpha=.35, palette=['r', 'b', 'g'])
        sns.histplot(data=logr, x=column, hue=hue, binrange=[-.5, .5], bins=100, stat='density', common_norm=False, alpha=.5, palette=colors, ax=ax)

        #breakpoint()

        sns.kdeplot(data=logr[logr[hue] == 'pSYMBA'], x=column, color=colors[0], bw_adjust=7.5, ax=ax, label='pSYMBA')
        sns.kdeplot(data=logr[logr[hue] == 'LSE'], x=column, color=[.5, .5, .5], bw_adjust=20, ax=ax, label='LSE')
        sns.kdeplot(data=logr[logr[hue] == self.LSEstock], x=column, color=colors[2], bw_adjust=7.5, ax=ax, label=self.LSEstock)
                
        x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
        x_pdf = np.linspace(x0, x1, 1000)
        y_pdf = scipy.stats.norm.pdf(x_pdf, scale=0.02)

        ax.plot(x_pdf, y_pdf, color='k', linestyle=':', lw=1.5, alpha=1, label='Normal distribution ($\sigma^2$=0.02)')
        ax.legend()
        
        ax.set_yscale('log')
        ax.set_xlim([-0.6,0.6])
        ax.set_ylim([10**-(4.5),10**2])
        #ax = sns.displot(data=logr, x=column, hue=hue, kind="kde", bw_adjust=.25, palette=['r', 'b'], facet_kws={'xlim': [-.5, .5]})
        plt.savefig(self.plotPath+'/Log_r_SYMBAfig4.pdf', dpi=100)
        plt.close()

    def price_buffer_returns(self):
        
        alphaLSE=.15
        color = [.5, 0, 0]

        fig = plt.figure(figsize=(9, 6.6))

        ax1 = plt.subplot(121)

        ax1.axvline(x=0, color='k', alpha=.3)
        ax1.axhline(y=0, color='k', alpha=.3)

        self.LSEstock = 'IHG'
        #self.LSEstock = 'DOM'
        #self.LSEstock = 'DLAR'
        #self.LSEstock = 'RSW'

        ax1.scatter(self.LSEr.mean(axis=0), self.LSEr.var(axis=0), color=[.6, .6, .6], alpha=alphaLSE) #, label='LSE (stock selection)')
        ax1.scatter(self.LSEr[self.LSEstock].iloc[:self.tInit].mean(), self.LSEr[self.LSEstock].iloc[:self.tInit].var(), color=color, alpha=1) #, label=self.LSEstock)
        
        ax1.scatter(self.LSEr.mean(axis=0).median(), self.LSEr.var(axis=0).median(), marker='x', color='k') #, label='LSE (grand mean)')

        ax1.set_xlim([-0.002, 0.002])
        ax1.set_ylim([-10**(-4.5), 0.004])

        ax1.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

        ax1.set_xlabel(r'$\mu$')
        ax1.set_ylabel(r'$\sigma^2$')

        ax2 = plt.subplot(122)

        ax2.axvline(x=0, color='k', alpha=.3)
        ax2.axhline(y=0, color='k', alpha=.3)

        ax2.scatter(self.LSEr.skew(axis=0), self.LSEr.kurt(axis=0), color=[.6, .6, .6], alpha=alphaLSE, label='LSE (stock selection)')

        ax2.scatter(self.LSEr.skew(axis=0).median(), self.LSEr.kurt(axis=0).median(), marker='x', color='k', label='LSE (grand median)')
        ax2.scatter(self.LSEr[self.LSEstock].iloc[:self.tInit].skew(), self.LSEr[self.LSEstock].iloc[:self.tInit].kurt(), color=color, alpha=1, label=self.LSEstock)

        #Lower bound K = Sk^2 + 1
        x = np.arange(ax2.get_xlim()[0], ax2.get_xlim()[1], 0.01)
        y = (x**2+1)-3
        ax2.plot(x, y, color='k', alpha=.1, linestyle='--', label='Kurtosis lower bound')

        ax2.set_xscale('symlog')
        ax2.set_yscale('symlog')

        ax2.set_xlabel('Skew[r]='+r'$\frac{{{\mu_3}}}{{{\sigma^3}}}$')
        ax2.set_ylabel('(Excess) Kurt[r]='+r'$\frac{{{\mu_4}}}{{{\sigma^4}}} - 3$')

        plt.tight_layout()
        fig.subplots_adjust(top=0.80)
        fig.legend( ncol=4, bbox_to_anchor=(0.87, 0.97) )
        
        plt.savefig(self.plotPathStylisedFacts+f'/SF_IHG.pdf', dpi=100)

        plt.close()

    def price_buffer_returns2(self):
        alphaLSE = 0.15
        color = [.5, 0, 0]

        self.LSEstock = 'IHG'

        fig = plt.figure(figsize=(7.5, 7))  # Adjust height to fit three subplots

        # Upper subplot (first row) - Time series plot
        ax0 = plt.subplot2grid((3, 2), (0, 0), colspan=2)  # Full width for this plot
        ax0.plot(self.LSE['IHG'].iloc[:self.tInit], color='k')
        ax0.set_xlabel('Trading day (t)')
        ax0.set_ylabel('Price (£)')
        #ax0.set_title('(a)', loc='left')  # Alphabetic label
        letter = string.ascii_lowercase[0]  
        ax0.text(-0.115, 1.0, letter, transform=ax0.transAxes, size=12, weight='bold')

        # First subplot (lower left) - Mean vs. Variance scatter plot
        ax1 = plt.subplot2grid((3, 2), (1, 0), rowspan=2)  # Bottom left plot
        ax1.axvline(x=0, color='k', alpha=.3)
        ax1.axhline(y=0, color='k', alpha=.3)
    
        ax1.scatter(self.LSEr.mean(axis=0), self.LSEr.var(axis=0), color=[.6, .6, .6], alpha=alphaLSE, label='LSE') #(stock selection)
        ax1.scatter(self.LSEr.mean(axis=0).median(), self.LSEr.var(axis=0).median(), marker='x', color='k', label='LSE (median)')
        ax1.scatter(self.LSEr[self.LSEstock].iloc[:self.tInit].mean(), self.LSEr[self.LSEstock].iloc[:self.tInit].var(), color=color, alpha=1, label=self.LSEstock)

        #print(self.LSEr[self.LSEstock].iloc[:self.tInit].var())
        
        ax1.set_xlim([-0.002, 0.002])
        ax1.set_ylim([-10**(-4.5), 0.004])
    
        ax1.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
        ax1.set_xlabel(r'$\mu$')
        ax1.set_ylabel(r'$\sigma^2$')
        #ax1.set_title('(b)', loc='left')  # Alphabetic label
        letter = string.ascii_lowercase[1]
        ax1.text(-0.25, 1, letter, transform=ax1.transAxes, size=12, weight='bold')  
        ax1.legend() 
        
        # Second subplot (lower right) - Skewness vs. Kurtosis scatter plot
        ax2 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)  # Bottom right plot
        ax2.axvline(x=0, color='k', alpha=.3)
        ax2.axhline(y=0, color='k', alpha=.3)
    
        ax2.scatter(self.LSEr.skew(axis=0), self.LSEr.kurt(axis=0), color=[.6, .6, .6], alpha=alphaLSE, label='LSE (stock selection)')
        ax2.scatter(self.LSEr.skew(axis=0).median(), self.LSEr.kurt(axis=0).median(), marker='x', color='k', label='LSE (grand median)')
        ax2.scatter(self.LSEr[self.LSEstock].iloc[:self.tInit].skew(), self.LSEr[self.LSEstock].iloc[:self.tInit].kurt(), color=color, alpha=1, label=self.LSEstock)
    
        # Kurtosis lower bound: K = Sk^2 + 1
        x = np.arange(ax2.get_xlim()[0], ax2.get_xlim()[1], 0.01)
        y = (x**2 + 1) - 3
        ax2.plot(x, y, color='k', alpha=.1, linestyle='--', label='Kurtosis lower bound')
    
        ax2.set_xscale('symlog')
        ax2.set_yscale('symlog')
        ax2.set_xlabel('Skew[r] = ' + r'$\frac{{{\mu_3}}}{{{\sigma^3}}}$')
        ax2.set_ylabel('(Excess) Kurt[r] = ' + r'$\frac{{{\mu_4}}}{{{\sigma^4}}} - 3$')
        #ax2.set_title('(c)', loc='left')  # Alphabetic label
        letter = string.ascii_lowercase[2]
        ax2.text(-0.15, 1, letter, transform=ax2.transAxes, size=12, weight='bold')  
        #ax2.legend()      
        
        plt.tight_layout()
        #fig.subplots_adjust(top=0.5)  # Adjust space for the title/legend
        #fig.legend(ncol=4, bbox_to_anchor=(0.5, 0.5))  # Adjust legend position
    
        plt.savefig(self.plotPathStylisedFacts + f'/SF_IHG_new.pdf', dpi=100)
        plt.close()


    def traders_dynamics(self):
        
        traderDynamicsFolder = '/Traders_dynamics'
        os.mkdir(self.plotPath+traderDynamicsFolder) if os.path.isdir(self.plotPath+traderDynamicsFolder) == False else None

        for s in range(self.S):

            plt.figure(figsize=(6, 7))

            ax1 = plt.subplot(411)
            ax1.plot(self.VolumeOwned[s].T)
            ax1.set_ylabel(r'Stocks owned')
            plt.tick_params('x', labelbottom=False)

            ax2 = plt.subplot(412, sharex=ax1)
            ax2.plot(self.Aequity[s].T)
            ax2.set_ylabel('Stock value (€)')
            plt.tick_params('x', labelbottom=False)

            ax3 = plt.subplot(413, sharex=ax1)
            ax3.plot(self.AvailableMargin[s].T)
            ax3.set_ylabel('Wealth (€)')
            plt.tick_params('x', labelbottom=False)

            ax4 = plt.subplot(414, sharex=ax1)
            ax4.plot(self.Atotal[s].T)
            ax4.set_ylabel('Total wealth (€)')
            ax4.set_xlabel('Time step')

            plt.tight_layout()

            plt.savefig(self.plotPath+traderDynamicsFolder+f'/simulation_{s}', dpi=100)

            plt.close()

    def parameter_analysis(self):

        s = 0

        X = np.zeros((self.I-1, 8)) #19
        
        parameter_list = ['kappa', r'$\rho_{bid}$', r'$\tau$', 'g', 'DdL', r'$\beta$', r'$\alpha_{+}$', r'$\alpha_{-}$'] #, r'$A_{total}$']

        #SYMBA
        X[:, 0] = self.kappa
        X[:, 1] = self.rhoBid
        X[:, 2] = self.tau
        X[:, 3] = self.g
        X[:, 4] = self.DrawdownLimit

        #RL
        X[:, 5] = self.beta
        X[:, 6] = self.alphaConf
        X[:, 7] = self.alphaDisc

        #RL variables

        #QF = self.fetch.saved('QFend', s=s, verbose=False)[0]
        #QT = self.fetch.saved('QTend', s=s, verbose=False)[0]

        #QT = np.sum(QT, axis=(1, 2, 3, 4, 5, 7))

        #X[:, 10] = QF[:, 0] #forecasting strategy
        #X[:, 11] = QF[:, 1] #forecasting strategy
        #X[:, 12] = QF[:, 2] #forecasting strategy

        #X[:, 13] = QT[:, 0] #bear or bull
        #X[:, 14] = QT[:, 1] #bear or bull
        #X[:, 15] = QT[:, 2] #bear or bull

        #Market variables
        #X[:, 16] = self.Aequity[s, -1]
        #X[:, 17] = self.Abonds[s, -1]
        #X[:, 18] = self.VolumeOwned[s, -1]
        #X[:, 10] = self.Atotal[s, -1]

        #X_transformed = TSNE(n_components=3).fit_transform(X)
        dr = PCA() #n_components=3
        X_transformed = dr.fit_transform(scale(X))
        
        fig = plt.figure(figsize=(9, 9))
        #ax = fig.add_subplot(111, projection='3d')
        #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=self.Atotal[s, -1], s=self.Qs[s, -1]/20+15), plt.colorbar()
        #p = ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=self.Atotal[s, -1], s=self.Qs[s, -1]/15+1)
        #ax.set_xlabel('PC1')
        #ax.set_ylabel('PC2')
        #ax.set_zlabel('PC3')
        #fig.colorbar(label='Total wealth (€)', pad=0.2)

        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=self.Atotal[s, -1], s=self.VolumeOwned[s, -1]/15+1)
        plt.xlabel('PC1')
        plt.ylabel('PC2')

        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(label='Total wealth (€)')
        plt.savefig(self.plotPath+'/Parameters_PCA', dpi=100)
        
        #print( dr.explained_variance_)
        #print( dr.explained_variance_ratio_ )
        #print( dr.explained_variance_ratio_.cumsum() )

        plt.figure()
        plt.bar(range(np.shape(dr.explained_variance_ratio_)[0]), dr.explained_variance_ratio_, color='k')
        plt.plot(dr.explained_variance_ratio_.cumsum(), color='k', alpha=.2)
        plt.savefig(self.plotPath+'/Pca_explained_variance', dpi=100)

        loadings = pd.DataFrame(data=dr.components_.T, index=parameter_list)
        #my_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(loadings))] 
        my_colors = cm.get_cmap('jet')(np.linspace(0, 1, len(loadings)))
        #plt.figure(figsize=(8, 5))
        loadings.abs().plot.bar(figsize=(8,5.8), color=my_colors)
        plt.ylabel('Absolute loadings')
        plt.legend(bbox_to_anchor=(1.1, 1.05))

        plt.savefig(self.plotPath+'/Pca_loadings', dpi=100)
        plt.close()

    def autocorrelation(self):

        plt.figure(figsize=(12, 12))

        for s in range(self.S):

            ax = plt.subplot(5, 4, s+1)
            tsaplots.plot_acf(self.rdf[s], ax=ax, title='Sim '+str(s), color='k', vlines_kwargs={'color': 'k', 'alpha': .3}, zero=False)
            plt.tick_params('x', labelbottom=False) if s < 15 else None
            ax.set(yticklabels=[]) if np.mod(s, 4) > 0 else None # remove the tick labels

        plt.tight_layout()
        plt.savefig(self.plotPath+'/Autocorrelations', dpi=100)
        plt.close()
        
    def brownian(self, n, delta, dt=1, x0=0, out=None):
        """
        Generate an instance of Brownian motion (i.e. the Wiener process):

            X(t) = X(0) + N(0, delta**2 * t; 0, t)

        where N(a,b; t0, t1) is a normally distributed random variable with mean a and
        variance b.  The parameters t0 and t1 make explicit the statistical
        independence of N on different time intervals; that is, if [t0, t1) and
        [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
        are independent.
    
        Written as an iteration scheme,

            X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)

        If `x0` is an array (or array-like), each value in `x0` is treated as
        an initial condition, and the value returned is a numpy array with one
        more dimension than `x0`.
    
        Arguments
        ---------
        x0 : float or numpy array (or something that can be converted to a numpy array
             using numpy.asarray(x0)).
            The initial condition(s) (i.e. position(s)) of the Brownian motion.    
        n : int
            The number of steps to take.
        dt : float
            The time step.
        delta : float
            delta determines the "speed" of the Brownian motion.  The random variable
            of the position at time t, X(t), has a normal distribution whose mean is
            the position at time t=0 and whose variance is delta**2*t.
        out : numpy array or None
            If `out` is not None, it specifies the array in which to put the
            result.  If `out` is None, a new numpy array is created and returned.

        Returns
        -------
        A numpy array of floats with shape `x0.shape + (n,)`.
    
        Note that the initial value `x0` is not included in the returned array.
        """
        x0 = np.asarray(x0)

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        r = norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

        # If `out` was not given, create an output array.
        if out is None:
            out = np.empty(r.shape)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples. 
        #np.cumsum(r, axis=-1, out=out)

        # Add the initial condition.
        out += np.expand_dims(x0, axis=-1)

        return out

    def AC(self):

        Tw = 5 #trading week-days
        Tm = 21 #monthly trading days
        Ty = 252 #yearly trading days

        #Deltat = [2*Tw, 3*Tm, Ty]
        Deltat = [1, Tm, Ty]
        Deltat_string = ['1d', '1m', '1Y']
        colors = [[.5, 0, 0], [.75, .75, .75], 'k']
        #colors = ['k', [.85, .85, .85], [.5, 0, 0]]
        linestyles = ['--', '-', '-'] #, ':']
        
        deltaBrownian = 0.02
        
        brownian = [self.brownian(n=len(self.rdf), delta=deltaBrownian) for _ in range(self.S*1000)]

        hue = 'Data'
        column = 'AC'#+str(deltat)

        fig, axs = plt.subplots(figsize=(9, 9), nrows=3, ncols=1, sharex=True)
               
        for deltan, (deltat, deltat_string) in enumerate(zip(Deltat, Deltat_string)):
            
            #[0::3] to skip 2 prices as in 3 auctions per day at LSE
            #sim = pd.DataFrame( data = self.rdf.apply(lambda x : sm.tsa.acf(x[0::3], nlags=deltat)[-1]), columns=[column])
            sim = pd.DataFrame( data = self.rdf.apply(lambda x : sm.tsa.acf(x, nlags=deltat)[-1]), columns=[column])
            LSE = pd.DataFrame( data = self.LSEr.apply(lambda x : sm.tsa.acf(x, nlags=deltat)[-1]), columns=[column] )
            #LSEstock = pd.DataFrame( data = self.LSEr[self.LSEstock].apply(lambda x : sm.tsa.acf(x, nlags=deltat)[-1]), columns=[column] )
            Wiener = pd.DataFrame( data = [sm.tsa.acf(b, nlags=deltat)[-1] for b in brownian], columns=[column] )
            
            sim_hue = f'pSYMBA ({deltat_string})'
            LSE_hue = f'LSE ({deltat_string})'
            Wiener_hue = f'Wiener ($\sigma^2={str(deltaBrownian)}$) ({deltat_string})'
            
            sim[hue] = sim_hue
            LSE[hue] = LSE_hue
            #LSEstock[hue] = f'{self.LSEstock} ({deltat_string})'
            Wiener[hue] = Wiener_hue
            
            data_hue_list = [sim_hue, LSE_hue, Wiener_hue]
                
            #data = pd.concat([sim, LSE], ignore_index=True)
            #sns.histplot(data=data, x=column, hue=hue, stat='density', element='poly', fill=False, common_norm=False, alpha=.35, palette=['r', 'b'], line_kws={'linestyle':'--'})
        
            ax = axs[deltan]
            #sns.histplot(data=data, x=column, hue=hue, common_norm=False, alpha=.2, fill=False, ax=ax, palette=[colors[1], colors[2]], binrange=[-.15, .15], bins=50, stat='density') # element='step'
            
            #sns.histplot(data=LSE, x=column, common_norm=False, element='step', alpha=.2, fill=False, ax=ax, color=colors[1], stat='density') #
            ax.axvline(x=sm.tsa.acf(self.LSEr[self.LSEstock], nlags=deltat)[-1], color=[0, 0, .5], label=self.LSEstock)
            ax.axvline(x=0, color=[.8, .8, .8], linestyle=':', linewidth=1.5)
            
            data = pd.concat([sim, LSE, Wiener], ignore_index=True)
            #breakpoint()
            #print(data.groupby('Data').max())
            #print()
            #p = sns.kdeplot(data=data, x=column, hue=hue, common_norm=False, palette=colors, fill=False, bw_adjust=2, ax=ax)
            
            for hue_n, data_hue in enumerate(data_hue_list):
                #breakpoint()
                sns.kdeplot(data=data[data[hue]==data_hue], x=column, common_norm=False, color=colors[hue_n], fill=False, bw_adjust=2, ax=ax, label=data_hue) #, alpha=.35) #, palette=['r', 'b']*3)
            
            #handles = p.legend_.legendHandles[::-1]

            #for line, ls, color, handle in zip(p.lines, linestyles, colors, handles):
            #    line.set_linestyle(ls)
                #line.set_linestyle(1.5) if 
            #    line.set_color(color)
            #    handle.set_ls(ls)
            #    handle.set_color(color)
            
            ax.legend()
            ax.set_xlim([-.45, .45])
            #ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim([10**-(3), 10**1.5])
            
        plt.tight_layout()
         
        #for deltat, deltat_string in zip(Deltat, Deltat_string):
        #    label = 'pSYMBA '+deltat_string
        #    ax = sns.kdeplot(data=data[data[hue]==label], x=column, hue=hue, common_norm=False, alpha=.35, palette=['r'], linestyle=linestyles[deltat_string], ax=ax, label=label)   
        #    label = 'LSE '+deltat_string
        #    ax = sns.kdeplot(data=data[data[hue]==label], x=column, hue=hue, common_norm=False, alpha=.35, palette=['b'], linestyle=linestyles[deltat_string], ax=ax, label=label)
        
        plt.savefig(self.plotPath+'/AC_SYMBAfig6.pdf', dpi=100)
        #plt.savefig(self.plotPath+'/AC_SYMBAfig6-mod3.pdf', dpi=100)
        plt.close()

    def AC_clustering(self):
    
        #from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
        for s in range(self.S):
            
            df = self.rdf[s].abs()
            
            pd.plotting.autocorrelation_plot(df)
            plt.xlim([1, 252])
            #plot_acf(df, lags=252)

            plt.savefig(self.plotPath+f'/AC_clustering_sim_{s}_acf.pdf', dpi=100)
            plt.close()

    def AC_volatility(self):

        Tw = 5 #trading week-days
        Tm = 21 #monthly trading days
        Ty = 252 #yearly trading days

        colors = [[.5, 0, 0], [.75, .75, .75], 'k']
        linestyles = ['--', '-', '-']
        
        deltaBrownian = 0.02
        
        #brownian = [self.brownian(n=len(self.rdf), delta=deltaBrownian) for _ in range(self.S*100)]       
        #brownian = pd.DataFrame(data=np.array(brownian).T)
        
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

    def stylised_facts(self):
        
        '''
        1. 1-lag return autocorrelation (should be close to uncertainty level).
        2. sum autocorrelation of |r_t| (abs return) from lag 1 to lag 100 (should be not zero). r_t = P_t - P_{t-1} ? or log? 
        3. Kurtosis - 3
        4. Hill estimator over the largest 2% of the price returns (to have quick computations). More sophisticated (but 1000 times slower): use the powerlaw package.
        5. Rcdf of |r_t|
        '''
 
        #### SF1 ####

        column = '1-Lag AC'
        SF1sim = pd.DataFrame( data = self.rdf.apply(lambda x : sm.tsa.acf(x, nlags=1)[1]), columns=[column])
        SF1LSE = pd.DataFrame( data = self.LSEr.apply(lambda x : sm.tsa.acf(x, nlags=1)[1]), columns=[column] )

        hue = 'Data origin'
        SF1sim[hue] = 'pSYMBA'
        SF1LSE[hue] = 'LSE'

        SF1 = pd.concat([SF1sim, SF1LSE])

        plt.figure()        
        sns.histplot(data=SF1, x=column, hue=hue, bins=50, stat='density', common_norm=False, alpha=.35, palette=['r', 'b'])
        plt.savefig(self.plotPathStylisedFacts+'/SF1', dpi=100)
        plt.close()

        #Alternative
        #N = len(SF1LSE)
        #plt.scatter(np.zeros(len(SF1)), SF1, color='k', alpha=.5, label='pSYMBA')
        #plt.scatter(0, SF1.mean(), color='r', marker='x', label=f'pSYMBA (grand mean S={self.S})')
        #plt.scatter(np.ones(N), SF1LSE, color='b', alpha=.05, label='LSE')
        #plt.scatter(1, SF1LSE.mean(), color='c', marker='x', label=f'LSE (grand mean N={N})')
        #plt.legend()


        #### SF2 ####

        column = r'$\Sigma$ 100-Lag $AC_{|r|}$'
        SF2sim = pd.DataFrame( data = self.rdf.apply(lambda x : (sm.tsa.acf(x.abs(), nlags=100)[1:]).sum()), columns=[column])
        SF2LSE = pd.DataFrame( data = self.LSEr.apply(lambda x : (sm.tsa.acf(x.abs(), nlags=100)[1:]).sum()), columns=[column] )

        hue = 'Data origin'
        SF2sim[hue] = 'pSYMBA'
        SF2LSE[hue] = 'LSE'

        SF2 = pd.concat([SF2sim, SF2LSE])

        plt.figure()        
        sns.histplot(data=SF2, x=column, hue=hue, stat='density', common_norm=False, alpha=.35, palette=['r', 'b'])
        plt.savefig(self.plotPathStylisedFacts+'/SF2', dpi=100)
        plt.close()

        #### SF3 ####

        column = r'Excess kurtosis'
        SF3sim = pd.DataFrame( data = self.rdf.kurt(axis=0), columns=[column])
        SF3LSE = pd.DataFrame( data = self.LSEr.kurt(axis=0), columns=[column] )

        hue = 'Data origin'
        SF3sim[hue] = 'Simulation'
        SF3LSE[hue] = 'LSE'
        
        WarningPrint = 'Negative Kurtosis not plotted' if (SF3sim[column]<0).sum() > 0 else None
        SF3sim = SF3sim[SF3sim[column]>=0] #!!! Discarding negative kurtosis for log scale plotting!

        SF3 = pd.concat([SF3sim, SF3LSE])

        plt.figure()        
        sns.histplot(data=SF3, x=column, hue=hue, stat='density', common_norm=False, alpha=.35, palette=['r', 'b'], log_scale=True).set(title=WarningPrint)
        
        plt.savefig(self.plotPathStylisedFacts+'/SF3', dpi=100)
        plt.close()
            
        #### SF5: R-cdf ####

        #X = np.arange(0, self.LSEr.abs().max().max()*1.1, 0.01)
        X = np.arange(0, .2, 0.0005)

        plt.figure(figsize=(9, 6))
        ax = plt.subplot(111)

        for s in range(self.S):
            
            Absr = self.rdf[s].abs().values
            y = np.array([(Absr >= x).sum()/len(self.rdf) for x in X])
            ax.step(X, y, color=[.5, 0, 0], alpha=.4, label='pSYMBA')
        #plt.yscale('symlog')

        N = 200
        for n in self.LSEr.columns[:N]:
            
            #SortedAbsr = sorted(self.LSEr[n].abs().values)
            Absr = self.LSEr[n].abs().values
            y = np.array([(Absr >= x).sum()/len(self.LSEr) for x in X])
            ax.step(X, y, color=[.5, .5, .5], alpha=.1, label='LSE')
        

        #Normal    
        Absr = abs(np.random.normal(scale=0.02, size=252*10))
        y = np.array([(Absr >= x).sum()/len(Absr) for x in X])
        ax.step(X, y, color=[0, 0, 0], alpha=.8, label='Normal '+r'$\sigma^2=0.02$')

        #LSE stock of reference
        #SortedAbsr = sorted(self.LSEr[self.LSEstock].abs().values)
        Absr = self.LSEr[self.LSEstock].abs().values
        y = np.array([(Absr >= x).sum()/len(self.LSEr) for x in X])
        ax.step(X, y, color=[0, 0, .6], alpha=1, label=self.LSEstock)

        ax.set_yscale('log')
                
        def legend_without_duplicate_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            leg = ax.legend(*zip(*unique))
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

        legend_without_duplicate_labels(ax)

        plt.xlabel(r'$|r|$')
        plt.ylabel('Cumulative frequency')
        #sns.histplot(data=SF2, x=column, hue=hue, alpha=.35, palette=['r', 'b'])
        plt.yscale('log')
        plt.savefig(self.plotPathStylisedFacts+'/SF5.pdf', dpi=100)
        plt.close()
        
        
    def forecast_preference(self):
    
        cmap = 'coolwarm_r'

        x = self.tau.values/max(self.tau)
        y = np.log(self.kappa.values)
        y -= min(y)
        y /= max(y)
            
        x *=100 #resolution
        y *=100

        z = np.zeros(( self.S, len(x) ))
        X = np.zeros_like(z)
        Y = np.zeros_like(z)
        
        for s in range(self.S):

            aFtemp = self.RL_variable('actionForecasting', s=s)
            z[s] = aFtemp[:, self.tInit:].mean(axis=1)
            X[s] = x.copy()
            Y[s] = y.copy()
            
        x = X.flatten()
        y = Y.flatten()
        z = z.flatten()
    
        for s in range(self.S): #range(1): #range(self.S):

            #QF = self.RL_variable('QForecasting')
            #aF = self.RL_variable('actionForecasting', s=s)

            #z = aF[:, self.tInit:].mean(axis=1)
       
            #y = np.log( self.Atotal[:, -100:].mean(axis=1) )
      
            #fig = plt.figure(figsize=(12, 6))
            
            plt.figure(figsize=(6, 6))
                
            #plt.subplot(131) 
            #axs[0].scatter(x, y, alpha=.2, c=z, cmap='coolwarm_r')
            #axs[0].xlabel(r'$\tau$')
            #axs[0].ylabel(r'$\kappa$')
            ##plt.colorbar()
            ##plt.ylabel('Forecasting strategy'+r'$(Q_1-Q_0)$')

            #cbar = plt.colorbar()
            #cbar.ax.get_yaxis().labelpad = 15
            #cbar.ax.set_ylabel('Fundamentalism (%)', rotation=270)

            grid_x, grid_y = np.mgrid[x.min():x.max()+1, y.min():y.max()+1]
            #grid_x, grid_y = np.mgrid[x.min():x.max(), y.min():y.max()]
            #breakpoint()
            z_interpolated = griddata((x,y), z, (grid_x, grid_y), method='linear')
            
            im = plt.imshow(z_interpolated, cmap=cmap, origin='lower', extent=(0, 1, 0, 1), vmin=0, vmax=1, aspect='auto')
            plt.xlabel(r'$\log{\kappa}$ (normalised)')
            plt.ylabel(r'$\tau$ (normalised)')            

            cbar = plt.colorbar(im)
            #cbar.set_label(r'$\bar{a_{\mathcal{F}}}$')
            cbar.ax.set_title(r'$\bar{a}_{\mathcal{F}}$')

            title = f'/forecast_preference_mesh_s{s}.pdf'
            #self.reporting(fig, title, report=report)
            plt.savefig(self.plotPath+title, dpi=320)
            
            plt.close()
        
    def forecast_preference_old(self, report=False):
 
        cmap = 'coolwarm_r'
        norm = mpl.colors.Normalize(vmin=0,vmax=100)
                
        for s in range(self.S):

            x = self.tau.values/max(self.tau)
            y = np.log(self.kappa.values)
            y -= min(y)
            y /= max(y)
            
            x *=100
            y *=100
    
            #QF = self.RL_variable('QForecasting')
            aF = self.RL_variable('actionForecasting', s=s)

            z = aF[:, self.tInit:].mean(axis=1)*100 #should be tInitLearn
       
            #y = np.log( self.Atotal[:, -100:].mean(axis=1) )
      
            #fig = plt.figure(figsize=(12, 6))
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 6))
            axs = axs.ravel()
                
            #plt.subplot(131) 
            #axs[0].scatter(x, y, alpha=.2, c=z, cmap='coolwarm_r')
            #axs[0].xlabel(r'$\tau$')
            #axs[0].ylabel(r'$\kappa$')
            ##plt.colorbar()
            ##plt.ylabel('Forecasting strategy'+r'$(Q_1-Q_0)$')

            #cbar = plt.colorbar()
            #cbar.ax.get_yaxis().labelpad = 15
            #cbar.ax.set_ylabel('Fundamentalism (%)', rotation=270)

            grid_x, grid_y = np.mgrid[x.min():x.max()+1, y.min():y.max()+1]
            #grid_x, grid_y = np.mgrid[x.min():x.max(), y.min():y.max()]
            #breakpoint()
            z_interpolated = griddata((x,y), z, (grid_x, grid_y), method='linear')
            
            #plt.subplot(131) 
            axs[0].imshow(z_interpolated, cmap=cmap, origin='lower', extent=(0, 100, 0, 100))
            

            #cbar = plt.colorbar()
            #cbar.ax.get_yaxis().labelpad = 15
            #cbar.ax.set_ylabel('Fundamentalism (%)', rotation=270)


            #plt.subplot(132)    
            y = aF.mean(axis=1)
            axs[1].scatter(self.tau, y, alpha=.2, c=z, cmap='coolwarm_r')
            axs[1].set_xlabel(r'$\tau$')
            axs[1].set_ylabel(r'$a_{\mathcal{F}}$')

            slope, intercept, r, p, se = stats.linregress(x, y)
            print('slope tau vs fund: ', slope, r, p)

            #unique_x = np.unique(x)        
            #plt.plot(unique_x, np.poly1d(np.polyfit(x, y, 1))(unique_x), color='k', linewidth=2)

            #plt.subplot(133)    
            axs[2].scatter(self.kappa, y, alpha=.2, c=z, cmap='coolwarm_r')
            axs[2].set_xlabel(r'$\kappa$')
            axs[2].set_ylabel(r'$a_{\mathcal{F}}$')
        
            axs[2].set_xscale('log')

            slope, intercept, r, p, se = stats.linregress(x, y)
            print('slope kappa vs fund: ', slope, r, p)
        
            slope, intercept, r, p, se = stats.linregress(np.log10(x), y)
            print('slope on log10: ', slope, r, p)
        
            X = np.zeros((np.shape(self.tau)[0], 2))
            X[:, 0] = self.tau
            X[:, 1] = np.log10(self.kappa)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
            model = LinearRegression().fit(X, y)
        
            print(model.coef_, model.score(X, y), )
        
            #unique_x = np.unique( np.log10(x) )        
            #plt.plot(unique_x, np.poly1d(np.polyfit( np.log10(x), y, 1))(unique_x), color='k', linewidth=2)
        
            #x_, y_ = np.meshgrid(x, y)
            #z_grid = np.array(z).reshape(1, -1)
            #breakpoint()
            #z_grid = np.ones((self.I, np.shape(z_grid)[0]))*z_grid
            #ax1 = plt.contourf(x_,y_,z_grid)

            plt.tight_layout()

            title = f'forecast_preference_new_s{s}.pdf' if not report else 'tau vs forecast preference'
            #self.reporting(fig, title, report=report)
            plt.savefig(self.plotPath+f'/forecast_preference_s{s}.pdf', dpi=100)
            
            plt.close()
            
    def QF_convergence_learning_mean(self):

        #print('Mean over specific traders only!')

        s = 0
        QF = self.RL_variable('QForecasting', s=s)

        T = range(self.T-self.tInit)
        #traders = range(50) #np.random.uniform(0, self.I, size=20).astype(int)
        traders = range(50) #np.random.uniform(0, self.I, size=20).astype(int)
        
        fig, axs = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
        axs = axs.flatten()

        for axn, ax in enumerate( axs ):

            ax.axhline(y=0, color=[.5, .5, .5], lw=0.5)

            [ ax.plot(T, np.squeeze( QF[trader, :, axn, 0] ), color='r', alpha=.02) for trader in traders ]
            [ ax.plot(T, np.squeeze( QF[trader, :, axn, 1] ), color='b', alpha=.02) for trader in traders ]

            #ax.plot(T, QF[traders, :, axn, 0].mean(axis=0) , color='r')
            #ax.plot(T, QF[traders, :, axn, 1].mean(axis=0) , color='b')
            
            #ax.plot(T, QF[:, :, axn, 0].mean(axis=0) , color='r')
            #ax.plot(T, QF[:, :, axn, 1].mean(axis=0) , color='b')
            
            #[ plt.plot(T, np.squeeze( QF[trader, :, 2] ), color='k', alpha=.02) for trader in traders ]
            ax.set_xlim([0, self.T])
            ax.set_ylim([QF.min(), QF.max()*1.05])
            #plt.ylim([-10, QF.max()*1.05])
            ax.set_xlabel(r'Trading day (t)') if axn >= 3 else None
            ax.set_ylabel(r'$Q_F$') if (axn % 3) == 0 else None
            
        #plt.legend()
        plt.savefig(self.plotPath+f'/convergence_QF_micro_mean_s{s}.pdf', dpi=120)

        plt.close()
        
    def forecasting_transition_matrix(self):
    
        s=0
    
        stateTrading = self.RL_variable('stateForecasting', s=s).astype(int)

        T = range(1, self.T-self.tInit)

        # Initialize the transition matrix
        num_states = 6
        transition_matrix = np.zeros((num_states, num_states))

        # Calculate the transition counts
        for t in T:
            for i in range(self.I):
                prev_state = stateTrading[i, t-1]
                curr_state = stateTrading[i, t]
                transition_matrix[prev_state, curr_state] += 1

        # Normalize the rows to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix_prob = np.divide(transition_matrix, row_sums, where=row_sums != 0)

        # Create a heatmap to visualize the transition matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(transition_matrix_prob, annot=True, fmt='.2f', cmap='viridis', cbar=True, ax=ax, vmin=0, vmax=1)

        # Add labels and title
        ax.set_xlabel(r'$s_{\mathcal{F}}(t+1)$')
        ax.set_ylabel(r'$s_{\mathcal{F}}(t)$')
        #ax.set_title(r'$s_{\mathcal{T}} \text{Transition Matrix}$')

        # Add a label to the color bar
        cbar = ax.collections[0].colorbar
        cbar.set_label(r'$\pi_{s_{t} \rightarrow s_{t+1}}$', fontsize=14)

        # Set ticks and labels for better readability
        ax.set_xticks(np.arange(num_states) + 0.5)
        ax.set_yticks(np.arange(num_states) + 0.5)
        
        
        s1Name = ['No trend & reverting gap',
        r'No trend & smaller gap$',
        r'No Trend & bigger gap',

        r'Trend & reverting gap',                
        r'Trend & smaller gap',
        r'Trend & bigger gap']
        
        ax.set_xticklabels(s1Name, rotation=30)
        ax.set_yticklabels(s1Name, rotation=30)

        # Adjust layout
        plt.tight_layout()

        # Save the plot as a PDF file
        plt.savefig(self.plotPath+'/transition_matrix_forecasting.pdf', format='pdf', dpi=300)        
        
