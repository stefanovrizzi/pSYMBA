#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:46:49 2022

@author: svrizzi
"""

import sys
sys.path.append('..') if '..' not in sys.path else None  #go to parent folder

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial']})
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
import string

import seaborn as sns

from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import pandas as pd
import scipy.stats
import os
import imageio.v2 as imageio

import gif
from tqdm import tqdm

# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

#[(plt.scatter(model.VolumeOwned[:, 0, t], model.FundamentalValues), plt.axhline(y=Price[0, -1]), plt.show()) for t in range(T)]

class PlotTraining:

    def __init__(self, fetch, s='training'):
        
        self.s = s
        self.fetch = fetch

        self.InputParameters = self.fetch.input_parameters()
        self.InputTimeParameters = self.fetch.input_parameters(time=True)
        self.trader_parameters() #to get tau
        self.input_parameters()
        self.market_parameters()
        self.market_variables()   

        self.create_folders()

    def create_folders(self):

        #Create folders
        try:
            os.mkdir(self.fetch.plotFolderPath) if os.path.isdir(self.fetch.plotFolderPath) == False else None #create Data folder for all data files, in case it does not exist
        except:
            self.plotFolderPath += '../'
            os.mkdir(self.plotFolderPath) if os.path.isdir(self.fetch.plotFolderPath) == False else None #create Data folder for all data files, in case it does not exist
        
        os.mkdir(self.fetch.plotFolderPath+self.fetch.example_folder_path()) if os.path.isdir(self.fetch.plotFolderPath+self.fetch.example_folder_path()) == False else None
        
        self.plotPath = self.fetch.plotFolderPath+self.fetch.example_folder_path()+f'/sim_{self.s}'
        os.mkdir(self.plotPath) if os.path.isdir(self.plotPath) == False else None #create Data folder for all data files, in case it does not exist
                
        self.giffAssemblyFolder = '/giff_assembly'

    def input_parameters(self):
    
        #Input parameters
        self.T = self.InputTimeParameters['Ttraining']
        self.tInit = self.fetch.parameters_json('SYMBA')['tInit'] #-100
        self.T -= self.tInit

        self.S = self.InputTimeParameters['Straining']
        
        self.Tbatch = np.array( self.InputTimeParameters['Tbatch'] )
        self.Tbatch -= self.tInit
        
        #self.T -= max(self.tau)         
        #self.Tbatch[0] -= max(self.tau)
        #self.T = self.Tbatch[:2].sum()
        
        self.I = self.InputParameters['I']
        self.J = self.InputParameters['J']
        self.traders = list(range(self.I))

        self.saveMicro = self.InputParameters['saveMicro']
        
        self.Tw, self.Tm, self.Ty = self.fetch.time_references()
        
    def market_parameters(self):
        
        marketParameters = self.fetch.parameters_json('market')
        
        self.tick = marketParameters['tick']        
        self.b = marketParameters['b']
        self.bFlat = marketParameters['bFlat']
        self.D = marketParameters['D']
        self.RiskFreeRate = marketParameters['RiskFreeRate']
        
        self.fundamentalValue = self.fetch.saved('fundamentalValue', "training_0")[:, 0, self.tInit:]

    def trader_parameters(self):

        #RL parameters
        self.RLparameters = self.fetch.parameters_csv('RL')
        self.CognitiveTraits = self.InputParameters['CognitiveTraits']
        
        self.beta = self.RLparameters['beta']
        self.alphaConf = self.RLparameters['alphaConf']
        
        #if 'optimistic bias' in self.CognitiveTraits:
        #    self.alphaDisc = self.RLparameters['alphaDisc']

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
        self.SYMBAparameters = self.fetch.parameters_csv('SYMBA')
        self.rhoAsk = self.SYMBAparameters['rhoAsk']
        self.rhoBid = self.SYMBAparameters['rhoBid']
        self.tau = self.SYMBAparameters['tau']
        self.g = self.SYMBAparameters['g']
        self.gSkewness = self.SYMBAparameters['gSkewness']
        self.kappa = self.SYMBAparameters['kappa']
        #self.ClosingLimit = self.SYMBAparameters['ClosingLimit']
        
        self.dfTraderParameters = pd.DataFrame()
        self.dfTraderParameters[r'$\beta$'] = self.beta
        self.dfTraderParameters[r'$\alpha_{+}$'] = self.alphaConf
        
        #if 'optimistic bias' in self.CognitiveTraits:
            #self.dfTraderParameters[r'$\alpha_{-}$'] = self.alphaDisc

        self.dfTraderParameters[r'$\alpha_{-}$'] = self.alphaDisc
            
        self.dfTraderParameters[r'$\rho_{ask}$'] = self.rhoAsk
        self.dfTraderParameters[r'$\rho_{bid}$'] = self.rhoBid
        self.dfTraderParameters[r'$\tau$'] = self.tau
        self.dfTraderParameters['g'] = self.g
        self.dfTraderParameters['gSkewness'] = self.gSkewness
        #self.dfTraderParameters['ClosingLimit'] = self.ClosingLimit

    def RL_variable(self, variableName, time=None):
       
       NactionsForecasting = 2 #3
       
       T = self.T if not time else 1
       
       if variableName == 'QForecasting':
          variable = np.zeros((self.I, T, 6, NactionsForecasting))
       elif variableName == 'QTrading':          
          variable = np.zeros((self.I, T, NactionsForecasting, 8, 3))
       elif variableName == 'QPricing':
          variable = np.zeros((self.I, T, 2, 2, 5))
       elif variableName == 'VPricing':
          variable = np.zeros((self.I, T, 2, 2))
       elif variableName == 'stateTrading':
          variable = np.zeros((self.I, T, 2))       
       elif variableName == 'actionTrading':
          variable = np.zeros((self.I, T))
       elif variableName == 'stateForecasting':
          variable = np.zeros((self.I, T))
       elif variableName == 'actionForecasting':
          variable = np.zeros((self.I, T))    
       if not time:
    
          for s, tbatch in enumerate(self.Tbatch[:]):
            
             folderName = f'training_{s}'
            
             tSkip = self.tInit
             #tSkip += max(self.tau)*(s<1)
            
             tbatchstart = self.Tbatch[:s].sum()*(s>0)
             tbatchend = tbatchstart+tbatch
                                
             if variableName in ['actionTrading', 'stateTrading', 'stateForecasting', 'actionForecasting']:
                variable[:, tbatchstart:tbatchend] = self.fetch.saved(variableName, folderName)[:, 0, tSkip:]
             else:
                variable[:, tbatchstart:tbatchend] = self.fetch.saved(variableName, folderName)[:, tSkip:]

       else:
             
             s = self.S-1
             
             folderName = f'training_{s}'
                                            
             if variableName == 'actionTrading':
                variable[:, 0] = self.fetch.saved(variableName, folderName)[:, 0, -time]
             else:
                variable[:, 0] = self.fetch.saved(variableName, folderName)[:, -time]

       return variable

    def market_variables(self):

       self.Price = np.zeros((self.J, self.T))
       self.Volume = np.zeros((self.J, self.T))
       self.BidAskSpread = np.zeros((self.J, self.T))

       self.r = np.zeros((self.J, self.T-1)) #returns

       self.fundamentalValueTrue = np.zeros((self.J, self.T))
       
       if self.saveMicro:
       
          self.VolumeOwned = np.zeros((self.I, self.J, self.T+1))
          self.Aequity = np.zeros((self.I, self.T+1))
          self.AvailableMargin = np.zeros((self.I, self.T+1))
          self.Wallet = np.zeros((self.I, self.T+1))
          self.Atotal = np.zeros((self.I, self.T+1))
 
          self.actionT = np.zeros((self.I, self.J, self.T))
       
       LSE = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)
       LSEstock = self.fetch.input_parameters()['LSEstock']
       self.PriceLSE = LSE[LSEstock].round(2)  
       self.PriceLSEInit = self.PriceLSE.iloc[:self.tInit] #.values[::-1] #GL9
       
       for s, tbatch in enumerate(self.Tbatch[:]):
            
            folderName = f'training_{s}'
                        
            tbatchstart = self.Tbatch[:s].sum()*(s>0)
            tbatchend = tbatchstart+tbatch
                    
            self.Price[:, tbatchstart:tbatchend] = self.fetch.saved('Price', folderName)[:, self.tInit:]
            self.Volume[:, tbatchstart:tbatchend] = self.fetch.saved('Volume', folderName)[:, self.tInit:]
            self.BidAskSpread[:, tbatchstart:tbatchend] = self.fetch.saved('BidAskSpread', folderName)[:, self.tInit:]

            self.r[:, tbatchstart:tbatchend-1] = np.log(self.Price[:, tbatchstart+1:tbatchend]) - np.log(self.Price[:, tbatchstart:tbatchend-1]) #returns
            
            self.fundamentalValueTrue[:, tbatchstart:tbatchend] = self.fetch.saved('fundamentalValueTrue', folderName)[:, self.tInit:] #true fundamental value

            if self.saveMicro:
            
                 self.VolumeOwned[:, :, tbatchstart:tbatchend+1] = self.fetch.saved('VolumeOwned', folderName)[:, :, self.tInit-1:]
                 self.Aequity[:, tbatchstart:tbatchend+1] = self.fetch.saved('Aequity', folderName)[:, 0, self.tInit-1:]
                 self.AvailableMargin[:, tbatchstart:tbatchend+1] = self.fetch.saved('AvailableMargin', folderName)[:, self.tInit-1:]
                 self.Wallet[:, tbatchstart:tbatchend+1] = self.fetch.saved('Wallet', folderName)[:, self.tInit-1:]
                 self.Atotal[:, tbatchstart:tbatchend+1] = self.fetch.saved('Atotal', folderName)[:, self.tInit-1:]
            
                 self.actionT[:, :, tbatchstart:tbatchend] = self.fetch.saved('actionTrading', folderName)[:, :, self.tInit:]

    ################################
    ############# Plots ############
    ################################

    def fundamental_value_convergence(self):
        
        print('needs saveMicro!')
        
        traders = np.random.choice(range(self.I), size=150)
        
        fundamentalValueTrue = self.fetch.saved('fundamentalValueTrue', s='training_0')[0, self.tInit:]
        fundamentalValue = self.fetch.saved('fundamentalValue', s='training_0')[:, 0, self.tInit:]
                
        # Compute the difference (absolute error) between each agent's signal and the reference signal
        
        # Normalize and color-code the kappa values
        norm = colors.LogNorm(vmin=np.min(self.kappa).round(4), vmax=np.max(self.kappa).round(2))
        col = cm.jet(norm(self.kappa))

        time = range(0, self.T, 25)

        # Create the plot
        fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
        axs = axs.flatten()

        # Plot each agent's error over time
        for i in traders:
            #breakpoint()
            differences = np.abs(fundamentalValue[i] / fundamentalValueTrue -1).round(3)*100
            #differences = np.log(fundamentalValue[i] ) - np.log( fundamentalValueTrue )
            axs[1].plot(time, differences[time], c=col[i], linewidth=.5, alpha=.5)

        # Plot the mean convergence for all agents
        #mean_difference = np.mean(differences, axis=0)
        #ax.plot(mean_difference, label='Mean Error', color='black', linewidth=2, linestyle='--')

        # Add labels, title, and legend
        axs[1].set_xlabel('Trading day (t)')
        axs[1].set_ylabel('Relative distance '+r'$\, \left| \frac{\Delta \tilde{f}}{f} \right| \, (\%) $')
        #ax.legend(loc='upper right', ncol=2)

        axs[1].set_yscale('log')
        axs[1].yaxis.set_major_formatter(ScalarFormatter())

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axs[1])
        #cbar.set_label(, fontsize=12) #\text{(log scale)}
        cbar.ax.set_title(r'$\kappa$')

        axs[0].plot(fundamentalValueTrue, label=r'$f$', color=[.5, .5, .5], zorder = 0)
        axs[0].plot(fundamentalValue[np.argmax(self.kappa), :], label=r'$\kappa = $'+f'{np.max(self.kappa).round(2)}', color=[.5, 0, 0], zorder = 1)
        axs[0].plot(fundamentalValue[np.argmin(self.kappa), :], label=r'$\kappa = $'+f'{np.min(self.kappa).round(4)}', color='b', zorder = 2)
        axs[0].set_xlabel('Trading day (t)')
        axs[0].set_ylabel('Fundamental value')
        axs[0].legend()

        for axn, ax in enumerate(axs):
            
            ax.text(-0.25, 1.0, string.ascii_lowercase[axn], transform=ax.transAxes, size=12, weight='bold')         

        plt.tight_layout()
        # Save the plot as a PDF file
        plt.savefig(self.plotPath+'/convergence_fundamental_value.pdf', format='pdf', dpi=300)    


    def reporting(self, fig, title, report=False):
    
        if not report:
            fig.savefig(self.plotPath+'/'+title, dpi=120)        
            plt.close()
        else:
            fig.suptitle(title)

    def inflation_t0(self):

        aF = self.RL_variable('actionForecasting')

        cmap = 'coolwarm_r'
        norm = mpl.colors.Normalize(vmin=0,vmax=100)

        plt.figure(figsize=(8, 6))

        plt.axvline(0, color=[.5, .5, .5], linewidth=.5, linestyle='--')
        plt.axhline(0, color=[.5, .5, .5], linewidth=.5, linestyle='--')

        plt.scatter( np.log(self.fundamentalValueTrue[0]/self.fundamentalValueTrue[0, 0]), np.log(self.Atotal.sum(axis=0)[1:]/self.Atotal.sum(axis=0)[0]), c=aF.mean(axis=0)*100, cmap=cmap, norm=norm, alpha=.5, s=3)

        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Fundamentalism (%)', rotation=270)
        
        plt.xlabel(r"$\log{f_t}-\log{f_{t_0}}$")
        plt.ylabel(r"$\log{W_t}-\log{W_{t_0}}$")
        
        plt.savefig(self.plotPath+f'/logEI_vs_logW_t0.png', dpi=300)
        plt.close()

    def inflation_t(self):

        aF = self.RL_variable('actionForecasting')

        cmap = 'coolwarm_r'
        norm = mpl.colors.Normalize(vmin=0,vmax=100)
        
        plt.figure(figsize=(8, 6))

        plt.axvline(0, color=[.5, .5, .5], linewidth=.5, linestyle='--')
        plt.axhline(0, color=[.5, .5, .5], linewidth=.5, linestyle='--')
        
        x = np.log(self.fundamentalValueTrue[0, 1:]/self.fundamentalValueTrue[0, :-1])
        #y = np.log(self.Atotal.sum(axis=0)[2:]/self.Atotal.sum(axis=0)[1:-1])
        y = np.log(self.Price[0, 1:]/self.Price[0, :-1])
        c = aF.mean(axis=0)[:-1]*100
        
        plt.scatter(x, y, c=c, cmap=cmap, norm=norm, alpha=.5, s=3)
        
        x_=x[c>90]
        y_=y[c>90]
                
        unique_x = np.unique(x_)
        
        plt.plot(unique_x, np.poly1d(np.polyfit(x_, y_, 1))(unique_x), color='k', linewidth=2)

        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Fundamentalism (%)', rotation=270)
        
        plt.xlabel(r"$\log{f_t}-\log{f_{t-1}}$")
        plt.ylabel(r"$\log{P_t}-\log{P_{t-1}}$")
        
        plt.savefig(self.plotPath+f'/logEI_vs_logP.png', dpi=300)
        plt.close()

    
    def QF_dynamics(self):
    
        self.QF = self.RL_variable('QForecasting')
        self.QFfolder = self.plotPath+'/QF'
        os.mkdir(self.QFfolder) if os.path.isdir(self.QFfolder) == False else None #create Data folder for all data files, in case it does not exist
    
        colours = ['k', 'r', 'b']

        traders = np.argsort(self.tau)
        n = min(25, self.I)            
        n_blocks = int(self.I/n)

        for block in range( n_blocks ):
        
            traders_temp = range( block*n, (block+1)*n ) 

            fig, axs = plt.subplots(int(np.sqrt(n)), int(np.ceil(np.sqrt(n))), figsize=(9, 9), sharex=True, sharey=True)
            axs = axs.ravel()
        
            for i, trader in enumerate(traders_temp):
                [axs[i].plot(self.QF[trader, :, a], alpha=.5, color=colours[a]) for a in range(np.shape(self.QF)[-1])]
  
            title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(self.QFfolder+f'/QF - traders {traders_temp[0]}-{traders_temp[-1]}.png', dpi=200)
            plt.close()
            
        del self.QF
        
        return

    def QT_dynamics(self):
 
        self.QT = self.RL_variable('QTrading')        
        self.QTfolder = self.plotPath+'/QT'
        os.mkdir(self.QTfolder) if os.path.isdir(self.QTfolder) == False else None #create Data folder for all data files, in case it does not exist
                
        colours = [[0, .5, 0], 'k', [.5, 0, 0]]
        
        traders = np.argsort(self.tau)
        n = min(25, self.I)            
        n_blocks = int(self.I/n)
        
        for s0 in range(np.shape(self.QT)[2]):
            for s1 in range(np.shape(self.QT)[3]):
                for s2 in range(np.shape(self.QT)[4]):
                                            
                    for block in range( n_blocks ):
        
                        traders_temp = range( block*n, (block+1)*n ) 

                        fig, axs = plt.subplots(int(np.sqrt(n)), int(np.ceil(np.sqrt(n))), figsize=(9, 9), sharex=True, sharey=True)
                        axs = axs.ravel()
                        
                        for i, trader in enumerate(traders_temp):
                            [axs[i].plot(self.QT[trader, :, s0, s1, s2, a], alpha=.8, color=colours[a]) for a in range(np.shape(self.QT)[-1])]              
                
                        title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
                        plt.suptitle(title)
                        plt.tight_layout()
                        plt.savefig(self.QTfolder+f'/QT training - {s0, s1, s2} - traders {traders_temp[0]}-{traders_temp[-1]}.png', dpi=200)
                        plt.close()
        
        del self.QT
        
        return
    
    def QP_dynamics(self):
    
        self.QP = self.RL_variable('QPricing')
        self.QPfolder = self.plotPath+'/QP'
        os.mkdir(self.QPfolder) if os.path.isdir(self.QPfolder) == False else None #create Data folder for all data files, in case it does not exist
        
        colours = [[.5, 0, 0], [0, .4, 0], [0, 0, .5]]
        
        traders = np.argsort(self.g)
        #traders = np.argsort(self.tau)
        n = min(25, self.I)            
        n_blocks = int(self.I/n)
        
        for s0 in range(np.shape(self.QP)[2]):
            for s1 in range(np.shape(self.QP)[3]):
                for s2 in range(np.shape(self.QP)[4]):
                    for s3 in range(np.shape(self.QP)[5]): 
                        for block in range( n_blocks ):
        
                            traders_temp = range( block*n, (block+1)*n )
                        
                            fig, axs = plt.subplots(int(np.sqrt(n)), int(np.ceil(np.sqrt(n))), figsize=(9, 9), sharex=True, sharey=True)
                            axs = axs.ravel()
                        
                            for i, trader in enumerate(traders_temp):
                                [axs[i].plot(self.QP[trader, :, s0, s1, s2, s3, a], alpha=.5, color=colours[a]) for a in range(np.shape(self.QP)[-1])]              
                
                        title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
                        plt.suptitle(title)
                        plt.tight_layout()
                        plt.savefig(self.QPfolder+f'/QP training - {s0, s1, s2, s3}.png', dpi=200)
                        plt.close()
        
        del self.QP
        
        return

    def trader_parameters_plot(self):
        
        #parameters = [r'$\alpha_{+}$', r'$\alpha_{-}$', r'$\tau$']
        parameters = [r'$\alpha_{+}$', r'$\alpha_{-}$']
        
        plt.figure(figsize=(12,12))
        #sns.pairplot(self.dfTraderParameters)
        
        sns.pairplot(self.dfTraderParameters[parameters], plot_kws={'color': 'k', 'alpha':0.05})
        plt.tight_layout()

        plt.savefig(self.plotPath+'/Trader_parameters', dpi=120)            

        plt.close()

    def QF_distribution(self):

        self.QF = self.RL_variable('QForecasting', time=1)        
        self.QFfolder = self.plotPath+'/QF'
        os.mkdir(self.QFfolder) if os.path.isdir(self.QFfolder) == False else None #create Data folder for all data files, in case it does not exist
        #breakpoint()
        #plt.scatter(self.QF[:, -1, 0], self.QF[:, -1, 1], c=self.tau), plt.show()
        palette = [[0, 0, .7], [.7, 0, 0], [.7, .7, .7]]
        
        idx = range(self.I)
                
        action = ['T', 'F'] #, 'M']
        Q = r'$Q_F$'
        a = r"$a_F$"
        
        index = pd.MultiIndex.from_product([idx, action], names = ["id", a])
        
        x = pd.DataFrame(self.QF[:, -1].reshape(-1), index=index, columns=[Q])

        xmin, xmax = x.min().values[0], x.max().values[0]
        #xmin, xmax = -10, 10
        
        #x.sort_values(by=[r"$s_1$", r"$s_0$"], ascending=[True, False], inplace=True)
        x.reset_index(inplace=True)
        #x = x.drop(x[(x[r'$a_T$'] == 'Ask')*(x[r'$s_0$']==s0[0])].index)
                                
        ax = sns.histplot(data=x.loc[idx], x=Q, stat='count', hue=a, multiple="dodge", palette=palette, alpha=.9, common_bins=True, common_norm=True, binrange=[xmin, xmax], bins=40, shrink=.9)
        ax.set_yscale('log')
                          
        plt.savefig(self.QFfolder+f'/QF final distribution.png', dpi=200)
        plt.close()
        
        del self.QF
        
        return

    def QT_distribution(self):

        self.QT = self.RL_variable('QTrading', time=1)        
        self.QTfolder = self.plotPath+'/QT'
        os.mkdir(self.QTfolder) if os.path.isdir(self.QTfolder) == False else None #create Data folder for all data files, in case it does not exist

        colours = [[0, .7, 0], 'k', [.8, 0, 0]] #g, k, r
        
        idx = range(self.I)
        
        s1 = [r'$V_{-} \, & \, H \leq P$',
        r'$V_{+} \, & \, P_{abp} \leq H \leq P$',
        r'$V_{+} \, & \, H \leq P_{abp} < P$',
        r'$V_{+} \, & \, H \leq P \leq P_{abp}$',
        
        r'$V_{-} \, & \, P < H$',
        r'$V_{+} \, & \, P_{abp} < P < H$',
        r'$V_{+} \, & \, P \leq P_{abp} \leq H$',
        r'$V_{+} \, & \, P < H < P_{abp} $']
        
        s1_order = [0, 2, 4, 6, 1, 3, 5, 7]        
        s1_dict = dict(zip(s1, s1_order))
        
        action = ['Bid', 'Hold', 'Ask']
        action_order = [0, 1, 2]
        action_dict = dict(zip(action, action_order))
        
        names = ["id", r"$s_1$", r"$a_T$"]
        
        index = pd.MultiIndex.from_product([idx, s1, action], names = names)
        
        x = pd.DataFrame(self.QT[:, -1].reshape(-1), index=index, columns=[r'$Q_T$'])

        #x[r'$Q_T$'] = x[r'$Q_T$']-x.min().values+1
        #xmin, xmax = np.log10(x.min().values[0]), np.log10(x.max().values[0])
        xmin, xmax = x.min().values[0], x.max().values[0]
        xmin = -1.1
        xmax = 1.1
        
        print('xmin fixed manually!')
        
        x.reset_index(inplace=True)
        x[r'$s1_{order}$'] = x[r"$s_1$"].map(s1_dict)
        x[r'$a_{order}$'] = x[r"$a_T$"].map(action_dict)
        x.sort_values(by=[r'$s1_{order}$', r'$a_{order}$'], ascending=[True, True], inplace=True)
        
        x = x.drop(x[(x[r'$a_T$'] == 'Ask')*(x[r'$s_1$']==s1[0])].index)
        x = x.drop(x[(x[r'$a_T$'] == 'Ask')*(x[r'$s_1$']==s1[4])].index)
                
        x.set_index([r'$s1_{order}$', r"$s_1$"], inplace=True)
            
        _, axs = plt.subplots(4, 2, figsize=(9, 8.5), sharex=True, sharey=True)
        axs = axs.flatten()
            
        for subplot, idx in enumerate(x.index.unique()): 
            
            #breakpoint()
            (s1_n, s1) = idx
                
            palette = colours if len( x.loc[idx][r"$a_T$"].unique() ) == len( x[r"$a_T$"].unique() ) else colours[:2]
                
            ax = sns.histplot(data=x.loc[idx], x=r'$Q_T$', stat='density', hue=r"$a_T$", multiple="dodge", palette=palette, alpha=.9, ax=axs[subplot], common_bins=True, common_norm=True, binrange=[xmin, xmax], bins=40, shrink=.95) # log_scale=True
            ax.set_yscale('log')
            ax.set_title(f'{s1}')
            None if subplot == 2 else ax.get_legend().set_visible(False)
                                      
        plt.tight_layout()

        plt.savefig(self.QTfolder+'/QT final distribution.pdf', dpi=200)
        plt.close()
        
        del self.QT
        
        return

    def QT_prob_distribution(self):

        self.QT = self.RL_variable('QTrading', time=1)        
        self.QTfolder = self.plotPath+'/QT'
        os.mkdir(self.QTfolder) if os.path.isdir(self.QTfolder) == False else None #create Data folder for all data files, in case it does not exist

        colours = [[0, .7, 0], 'k', [.8, 0, 0]] #g, k, r
        
        idx = range(self.I)

        s0 = [r'$T$', r'$F$'] #, r'$M$']

        s0_order = [0, 1, 2]        
        s0_dict = dict(zip(s0, s0_order))
        
        s1 = [r'$V_{-} \, & \, H \leq P$',
        r'$V_{+} \, & \, P_{abp} \leq H \leq P$',
        r'$V_{+} \, & \, H \leq P_{abp} < P$',
        r'$V_{+} \, & \, H \leq P \leq P_{abp}$',
        
        r'$V_{-} \, & \, P < H$',
        r'$V_{+} \, & \, P_{abp} < P < H$',
        r'$V_{+} \, & \, P \leq P_{abp} \leq H$',
        r'$V_{+} \, & \, P < H < P_{abp} $']
        
        s1_order = [0, 2, 4, 6, 1, 3, 5, 7]        
        s1_dict = dict(zip(s1, s1_order))
        
        action = ['Bid', 'Hold', 'Ask']
        action_order = [0, 1, 2]
        action_dict = dict(zip(action, action_order))
        
        names = ["id", r"$s_0$", r"$s_1$", r"$a_T$"]
        Qcolumn = r'$Q_T$'
        aColumn = r'$a_T$'
        
        index = pd.MultiIndex.from_product([idx, s0, s1, action], names = names)
        
        x = pd.DataFrame(self.QT[:, -1].reshape(-1), index=index, columns=[Qcolumn])

        x.reset_index(inplace=True)
        x[r'$s0_{order}$'] = x[r"$s_0$"].map(s0_dict)
        x[r'$s1_{order}$'] = x[r"$s_1$"].map(s1_dict)
        x[r'$a_{order}$'] = x[r"$a_T$"].map(action_dict)
        x.sort_values(by=[r'$s0_{order}$', r'$s1_{order}$', r'$a_{order}$'], ascending=[True, True, True], inplace=True)
        
        x['beta'] = x.reset_index().id
        x['beta'] = x.beta.map(dict(zip(x.beta.unique(), self.beta)))
        x[Qcolumn] *= x.beta
        x.drop('beta', axis=1, inplace=True)
                        
        x['exp'] = np.exp(x[Qcolumn])
        x.drop(Qcolumn, axis=1, inplace=True)
        x.reset_index(inplace=True)
        x = x.drop(x[(x[aColumn] == 'Ask')*(x[r'$s_1$']==s1[0])].index)
        x = x.drop(x[(x[aColumn] == 'Ask')*(x[r'$s_1$']==s1[4])].index)
        x.set_index(["id", r'$s0_{order}$', r"$s_0$", r'$s1_{order}$', r"$s_1$", aColumn], inplace=True)

        df = x.unstack(aColumn)['exp']

        df = df.apply(lambda x: x/df.sum(axis=1))
        df = df.stack(aColumn)
        df = df.to_frame().rename(columns={0: r'$\pi$'})
        df.reset_index(inplace=True)
        #breakpoint()
        df[r'$a_{order}$'] = df[aColumn].map(action_dict)
        df.sort_values(by=[r'$s0_{order}$',  r'$s1_{order}$', r'$a_{order}$'], ascending=[True, True, True], inplace=True)

        xmin = 0
        xmax = 1
        
        df.set_index([r'$s0_{order}$', r"$s_0$"], inplace=True)
        
        for index in df.index.unique():
        
            (s0n, s0) = index
        
            df_temp = df.loc[index]
            df_temp.reset_index(inplace=True)
            df_temp.set_index([r'$s0_{order}$', r"$s_0$", r'$s1_{order}$', r"$s_1$"], inplace=True)

            _, axs = plt.subplots(4, 2, figsize=(9, 8.5), sharex=True, sharey=True)
            axs = axs.flatten()
        
            for subplot, idx in enumerate(df_temp.index.unique()): 
            
                (s0n, s0, s1n, s1) = idx
                
                palette = colours if len( df_temp.loc[idx][aColumn].unique() ) == len( action ) else colours[:2]
                
                ax = sns.histplot(data=df_temp.loc[idx], x=r'$\pi$', stat='count', hue=aColumn, multiple="dodge", palette=palette, alpha=.9, ax=axs[subplot], common_bins=True, common_norm=True, binrange=[xmin, xmax], bins=50, shrink=.95) # log_scale=True
                ax.set_yscale('log')
                ax.set_title(f'{s1}')
                None if subplot == 2 else ax.get_legend().set_visible(False)
                                      
            plt.tight_layout()

            plt.savefig(self.QTfolder+f'/QT probability final distribution - {s0}.pdf', dpi=200)
            plt.close()
        
        del self.QT
        
        return

    def QP_distribution(self):
        
        QP = self.RL_variable('QPricing', time=1)        
        QPfolder = self.plotPath+'/QP'
        os.mkdir(QPfolder) if os.path.isdir(QPfolder) == False else None #create Data folder for all data files, in case it does not exist
                
        palette = [[.5, .0, .0], [.9, .0, .3], [0, .0, .5], [0, .3, .9], [.65, .65, .65]]
        
        idx = range(self.I)
        
        s0 = ['Bid', 'Ask']
        s1 = [r'$\sigma_{l}^2$', r'$\sigma_{H}^2$']
        
        Q = r'$Q_P$'
        a = r'$a_T$'
        
        action = [r'$P_{bid}+g$', r'$P_{bid}-\frac{g}{2}$', r'$P_{ask}+g$', r'$P_{ask}-\frac{g}{2}$', r'$M$']
       
        action_order = {r'$P_{bid}+g$':1, r'$P_{bid}-\frac{g}{2}$':2, r'$P_{ask}+g$':3, r'$P_{ask}-\frac{g}{2}$':4, r'$P_{bid}-g$': 6, r'$P_{bid}+\frac{g}{2}$':5, r'$P_{ask}-g$':8, r'$P_{ask}+\frac{g}{2}$':7, r'$M$':9}
                
        index = pd.MultiIndex.from_product([idx, s0, s1, action], names = ["id", r"$s_0$", r"$s_1$", a])
        x = pd.DataFrame(QP[:, -1].reshape(-1), index=index, columns=[Q])

        #x[Q] = x[Q]-x.min().values+1
        xmin, xmax = x.min().values[0], x.max().values[0]
        xmax = max(abs(xmax), abs(xmin))
        xmin = -max(abs(xmax), abs(xmin))
        #xmin, xmax = np.log10(x.min().values[0]), np.log10(x.max().values[0])

        x.reset_index(inplace=True)
        #x = x.drop(x[(x[a] == r'$M$')].index) #exclude market order, as value is 0

        gSkewDict = dict( zip(x.id.unique(), self.gSkewness) ) 
        x[r'$g_{skew}$'] = x.id.map(gSkewDict)
        x[r'$g_{skew}$'] = x[r'$g_{skew}$'].astype(int)

        reverseDict = {r'$P_{bid}+g$':r'$P_{bid}-g$', r'$P_{bid}-\frac{g}{2}$':r'$P_{bid}+\frac{g}{2}$', r'$P_{ask}+g$':r'$P_{ask}-g$', r'$P_{ask}-\frac{g}{2}$':r'$P_{ask}+\frac{g}{2}$', r'$M$':r'$M$'}   

        conditionBid = x[r"$s_0$"]=='Bid'
        conditionG = (x[r'$g_{skew}$'] > 0)

        x[a] = np.where(conditionBid*conditionG, x[a].map( reverseDict ), x[a])

        conditionAsk = x[r"$s_0$"]=='Ask'
        conditionG = (x[r'$g_{skew}$'] < 0)

        x[a] = np.where(conditionAsk*conditionG, x[a].map( reverseDict ), x[a])
                        
        x['action_order'] = x[a].map(action_order)
                
        subplotCol = len( x[r'$g_{skew}$'].unique() )
        
        x.sort_values(by=[r"$s_0$", r"$s_1$", r'$g_{skew}$', 'action_order'], inplace=True)
                      
        ##for nfig, fig in enumerate(x[r"$s_2$"].unique()):
        x.set_index([r"$s_0$", r'$g_{skew}$', r"$s_1$"], inplace=True)
 
        _, axs = plt.subplots(int( len(x.index.unique())/subplotCol ), subplotCol, figsize=(12, 8.5), sharex=True, sharey=True)
        axs = axs.flatten()
            
        for subplot, idx in enumerate(x.index.unique()): 

            (s0, g, s1) = idx
            
            ax = sns.histplot(data=x.loc[idx], x=Q, stat='count', hue=a, multiple="dodge", palette=palette, alpha=.9, ax=axs[subplot], common_bins=True, common_norm=True, binrange=[xmin, xmax], bins=20, shrink=.95)
            ax.set_yscale('log')
            ax.set_title(f'{s1} & {s0} '+r'$(g_{skew}=$'+f'{g})')
            None if subplot in [0, 1, 4, 5] else ax.get_legend().set_visible(False)
                          
        plt.tight_layout()
        plt.savefig(QPfolder+f'/QP final distribution.pdf', dpi=200)
        plt.close()
        
        del QP
        
        return

    def QP_prob_distribution(self):
        
        QP = self.RL_variable('QPricing', time=1)        
        QPfolder = self.plotPath+'/QP'
        os.mkdir(QPfolder) if os.path.isdir(QPfolder) == False else None #create Data folder for all data files, in case it does not exist
                
        palette = [[.5, .0, .0], [.9, .0, .3], [0, .0, .5], [0, .3, .9], [.65, .65, .65]]
        
        idx = range(self.I)
        
        s0 = ['Bid', 'Ask']
        s1 = [r'$\sigma_{l}^2$', r'$\sigma_{H}^2$']
        
        Q = r'$Q_P$'
        a = r'$a_P$'
        
        action = [r'$P_{bid}+g$', r'$P_{bid}-\frac{g}{2}$', r'$P_{ask}+g$', r'$P_{ask}-\frac{g}{2}$', r'$M$']
       
        action_order = {r'$P_{bid}+g$':1, r'$P_{bid}-\frac{g}{2}$':2, r'$P_{ask}+g$':3, r'$P_{ask}-\frac{g}{2}$':4, r'$P_{bid}-g$': 6, r'$P_{bid}+\frac{g}{2}$':5, r'$P_{ask}-g$':8, r'$P_{ask}+\frac{g}{2}$':7, r'$M$': 9}
                
        index = pd.MultiIndex.from_product([idx, s0, s1, action], names = ["id", r"$s_0$", r"$s_1$", a])
        x = pd.DataFrame(QP[:, -1].reshape(-1), index=index, columns=[Q])

        x.reset_index(inplace=True)
                                
        x['beta'] = x.id.map(dict(zip(x.id.unique(), self.beta)))
        x[Q] *= x.beta
        x.drop('beta', axis=1, inplace=True)

        x['exp'] = np.exp(x[Q])
        x.drop(Q, axis=1, inplace=True)
        x.reset_index(inplace=True)
        x.set_index(["id", r"$s_0$", r"$s_1$", a], inplace=True)

        df = x.unstack(a)['exp']

        df = df.apply(lambda x: x/df.sum(axis=1))
        df = df.stack(a)
        pi = r'$\pi$'
        df = df.to_frame().rename(columns={0: pi})
        df.reset_index(inplace=True)
        #breakpoint()

        gSkewDict = dict( zip(df.id.unique(), self.gSkewness) ) 
        df[r'$g_{skew}$'] = df.id.map(gSkewDict)
        df[r'$g_{skew}$'] = df[r'$g_{skew}$'].astype(int)

        reverseDict = {r'$P_{bid}+g$':r'$P_{bid}-g$', r'$P_{bid}-\frac{g}{2}$':r'$P_{bid}+\frac{g}{2}$', r'$P_{ask}+g$':r'$P_{ask}-g$', r'$P_{ask}-\frac{g}{2}$':r'$P_{ask}+\frac{g}{2}$', r'$M$':r'$M$'}   

        conditionBid = df[r"$s_0$"]=='Bid'
        conditionG = (df[r'$g_{skew}$'] > 0)
        df[a] = np.where(conditionBid*conditionG, df[a].map( reverseDict ), df[a])

        conditionAsk = df[r"$s_0$"]=='Ask'
        conditionG = (df[r'$g_{skew}$'] < 0)
        df[a] = np.where(conditionAsk*conditionG, df[a].map( reverseDict ), df[a])
                        
        df['action_order'] = df[a].map(action_order)

        subplotCol = len( df[r'$g_{skew}$'].unique() )

        df.sort_values(by=[r"$s_0$", r"$s_1$", r'$g_{skew}$', 'action_order'], inplace=True)
        
        df.set_index([r"$s_0$", r'$g_{skew}$', r"$s_1$"], inplace=True)

        xmin, xmax = 0, 1
        
        _, axs = plt.subplots(int( len(df.index.unique())/subplotCol ), subplotCol, figsize=(12, 8.5), sharex=True, sharey=True)
        axs = axs.flatten()
            
        for subplot, idx in enumerate(df.index.unique()): 

            (s0, g, s1) = idx
            
            ax = sns.histplot(data=df.loc[idx], x=pi, stat='count', hue=a, multiple="dodge", palette=palette, alpha=.9, ax=axs[subplot], common_bins=True, common_norm=True, binrange=[xmin, xmax], bins=20, shrink=.95)
            ax.set_yscale('log')
            ax.set_title(f'{s1} & {s0} '+r'$(g_{skew}=$'+f'{g})')
            None if subplot in [0, 1, 4, 5] else ax.get_legend().set_visible(False)
                          
        plt.tight_layout()
        plt.savefig(QPfolder+f'/QP probability final distribution.pdf', dpi=200)
        plt.close()
        
        del QP
        
        return


    # Price dynamics
    def evolutionary_trading(self, report=False):
           
        actionTrading = self.RL_variable('actionTrading')
        x = abs(actionTrading-1).mean(axis=0)*100

        x_smooth = pd.Series(x).rolling(100).mean()
        
        fig = plt.figure(figsize=(6, 4))

        #Price dynamics
        plt.plot(x, color='k')
        plt.plot(x_smooth, color='r', label='MA')
        plt.xlabel('Trading day (t)')
        #plt.ylabel(r'$\bar{a}_{\mathcal{T}}$')
        plt.ylabel('% agents trading')        
        plt.legend()

        title = f'evolutionary_training.pdf' if not report else 'Evolutionary trading'
        self.reporting(fig, title, report=report)
            
    # Price dynamics
    def price_dynamics(self, report=False):
           
        for j in range(self.J):

            fig = plt.figure(figsize=(10, 6))

            #Price dynamics
            plt.plot(self.Price[j, :], color='k', label='pSYMBA')
            plt.plot(self.fundamentalValueTrue[j, :], color='k', alpha=0.3, label=r'$x$')
            plt.xlabel('Time (days)')
            plt.ylabel('Stock price (€)')

            title = f'Price_dynamics_stock{j}_training.png' if not report else 'Market dynamics (training)'
            self.reporting(fig, title, report=report)

    # Price dynamics
    def market_dynamics(self, report=False):
        
        for j in range(self.J):

            fig, axs = plt.subplots(4, 1, figsize=(10, 7), sharex=True, sharey=False)
            axs = axs.ravel()

            #Price dynamics
            ax1 = axs[0]
            ax1.plot(self.Price[j, :], color='k', label='pSYMBA')
            ax1.plot(self.fundamentalValueTrue[j, :], color='k', alpha=0.3, label=r'$x$')
            timeLSE = min(len(self.PriceLSE)-self.tInit, self.T)
            ax1.plot(range(-self.tInit, timeLSE), self.PriceLSE.iloc[:timeLSE+self.tInit].values, color=[0, 0, .5], alpha=0.3, label='LSE')
            #ax1.axhline(y=self.FundamentalValues.mean(), color='r', label='Mean FV')
            #ax1.legend()
            
            #for i in range(self.I):
            #    ax1.plot( self.FundamentalValues[i]+self.g[i]*(self.Price[j, :]), color='b', alpha=.1)
            #    ax1.plot( self.FundamentalValues[i]-self.g[i]*(self.Price[j, :]), color='r', alpha=.1)
            
            #ax1.plot(self.FundamentalValues.mean()+max(self.g)*self.BidAskSpread[j, 20:], color='r', alpha=.5)
            #ax1.plot(self.FundamentalValues.mean()-max(self.g)*self.BidAskSpread[j, 20:], color='r', alpha=.5)
            #ax1.set_xlabel('Time step')
            ax1.set_ylabel('Stock price (€)')
            plt.tick_params('x', labelbottom=False)

            #Trading volume
            ax2 = axs[1]
            ax2.plot(self.Volume[j, :], color='k')
            #ax2.set_xlabel('Time step')
            ax2.set_ylabel('Traded volume')
            #ax2.set_yscale('symlog')
            plt.tick_params('x', labelbottom=False)

            #Bid-ask spread
            ax3 = axs[2]
            ax3.plot(self.BidAskSpread[j, :], color='k', alpha=.5)
            #ax3.set_xlabel('Time step')
            ax3.set_ylabel('Bid-Ask spread (€)')
            ax3.set_ylim([self.tick/2, None])
            ax3.set_yscale('symlog') #, linear_width=tick, subs=[-2, -1, 0, 1])
            plt.tick_params('x', labelbottom=False)
            #plt.savefig(self.plotPath+f'/Trading_volume_dynamics_stock{j}', dpi=120)

            ax4 = axs[3]
            sliding_window_view = np.lib.stride_tricks.sliding_window_view

            ax4.plot(sliding_window_view(self.r[j, :], window_shape = 5).var(axis=1), color='k')
            ax4.set_xlabel('Time step (trading days)')
            ax4.set_ylabel('Volatility ( Var[r] )')
            #ax4.set_yscale('log')
            #plt.savefig(self.plotPath+f'/Volatility_stock{j}', dpi=120)
            ax4.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)
            
            title = f'Market_dynamics_stock{j}_training.pdf' if not report else 'Market dynamics (training)'
            self.reporting(fig, title, report=report)
            
        return fig

    def price_volume_relationship(self):
    
        my_colors = cm.get_cmap('jet')(np.linspace(0, 1, self.T))

        j = 0
        plt.figure(figsize=(10, 7))
        plt.scatter(self.Price[j, :], self.Volume[j, :], alpha=.5, color=my_colors)
        plt.xlabel('Stock price')
        plt.ylabel('Volume')
        plt.savefig(self.plotPath+f'/Price_volume_relation_{j}_training.png', dpi=120)            
        plt.close()

    def returns_volume_change_relationship(self):

        my_colors = cm.get_cmap('jet')(np.linspace(0, 1, self.T-1))

        j = 0
        plt.figure(figsize=(10, 7))
        plt.scatter(self.r[j, :], self.Volume[j, 1:]-self.Volume[j, :-1], alpha=.5, color=my_colors)
        plt.xlabel('Stock price returns')
        plt.ylabel('Volume change')
        plt.savefig(self.plotPath+f'/Returns_volume_change_relation_{j}_training.png', dpi=120)            
        plt.close()

    def zero_returns_trading_relationship(self):
    
        actionTrading = self.RL_variable('actionTrading')

        my_colors = cm.get_cmap('jet')(np.linspace(0, 1, self.T-1))

        j = 0
        plt.figure(figsize=(10, 7))
        plt.scatter(self.r[j, :], (abs(actionTrading[:, 1:]-1)>0).sum(axis=0), alpha=.5, color=my_colors)
        plt.xlabel('Stock price returns')
        plt.ylabel('Trading')
        plt.savefig(self.plotPath+f'/Returns_trading_relation_{j}_training.png', dpi=120)            
        plt.close()

    def return_correlation(self):
        
        my_colors = cm.get_cmap('jet')(np.linspace(0, 1, self.T-1))

        for j in range(self.J):
            #Show correlation of jumps
            plt.figure(figsize=(7, 8))

            plt.subplot(211)
            plt.plot(self.Price[j, :-1]-self.Price[j, :].mean(), self.Price[j, 1:]-self.Price[j, :].mean(), color='k', alpha=.1)
            plt.scatter(self.Price[j, :-1]-self.Price[j, :].mean(), self.Price[j, 1:]-self.Price[j, :].mean(), color=my_colors)
            plt.xlabel('Stock price (t)')
            plt.ylabel('Stock price (t+1)')

            plt.subplot(212)
            plt.plot(self.r[j, :-1], self.r[j, 1:], color='k', alpha=.1)
            plt.scatter(self.r[j, :-1], self.r[j, 1:], color=my_colors[:-1])
            plt.xlabel('Return (t)')
            plt.ylabel('Return (t+1)')

            plt.tight_layout()

            plt.savefig(self.plotPath+f'/Price_(zero_mean)_and_returns_correlation_stock{j}', dpi=120)

            plt.close()
            
    #def check_actionT_vs_QT(self):
    
    #    plt.scatter((self.actionT[:, 0, :-1, 0] == -1).sum(axis=1), self.QT[:, -1, :, :, :, 0, :].sum(axis=(1, 2, 3, 4)), label='buy')
    #    plt.scatter((self.actionT[:, 0, :-1, 0] == 0).sum(axis=1), self.QT[:, -1, :, :, :, 1, :].sum(axis=(1, 2, 3, 4)), label='hold')
    #    plt.scatter((self.actionT[:, 0, :-1, 0]==1).sum(axis=1), self.QT[:, -1, :, :, :, 2, :].sum(axis=(1, 2, 3, 4)), label='sell'), plt.legend()
    #    plt.show()


    def attempts_vs_wealth(self, report=False):
    
        cmap = 'turbo'

        fig = plt.figure(figsize=(9, 9))
        
        x = (abs(self.actionT[:, 0, :-1]-1) > 0).sum(axis=1)
        y = self.Atotal[:, -1] #self.Wallet[:, -1]
        s = np.log(self.VolumeOwned[:, 0, -1]+10)*10
        color = (abs(self.VolumeOwned[:, 0, 1:]-self.VolumeOwned[:, 0, :-1]) > 0).sum(axis=1)

        plt.scatter(x, y, s=s, alpha=.4, c=color, cmap=cmap)
        plt.title('size = stocks owned, color = resolved trading - wealthiest trader: '+str(np.argmax(self.Atotal[:, -1])))
        plt.xscale('symlog', base=2)
        plt.yscale('symlog', base=10)
        plt.xlabel('Trading actions taken')
        plt.ylabel('Total final wealth ($)')
        
        title = f'Attempts_vs_wealth.pdf' if not report else 'Trading attempts vs Final wealth (training)'
        self.reporting(fig, title, report=report)
        
        
    def trading_attempts(self):
    
        cmap = 'turbo'

        fig = plt.figure(figsize=(12, 12))
        
        #breakpoint()
        
        x = (abs(self.actionT[:, 0, :-1]-1) > 0).sum(axis=1)
        y = (abs(self.VolumeOwned[:, 0, 1:]-self.VolumeOwned[:, 0, :-1]) > 0).sum(axis=1)
        #s = np.log10(self.Atotal[:, -1])
        s = self.Atotal[:, -1] / 2000000

        plt.subplot(321)
        #plt.scatter(x, y, s=s, alpha=.4, c=self.FundamentalValues, cmap=cmap)
        #plt.title('Fundamental Values')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        plt.subplot(322)
        plt.scatter(x, y, s=s, alpha=.4, c=self.g, cmap=cmap)
        plt.title(r'$g$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        plt.subplot(323)
        plt.scatter(x, y, s=s, alpha=.4, c=self.tau, cmap=cmap)
        plt.title(r'$\tau$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        plt.subplot(324)
        
        self.QF = self.RL_variable('QForecasting')
        
        #plt.scatter(x, y, s=s, alpha=.4, c=self.QF[:, -1, 1]-self.QF[:, -1, 0], cmap=cmap)
        plt.title('Forecasting strategy '+r'$Q_1 - Q_0$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        del self.QF
        
        try:
            
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, c=self.alphaConf-self.alphaDisc, cmap=cmap)
            plt.title(r'$\alpha$'+' bias')
            
        except:
            
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, c=self.alphaConf, cmap=cmap)
            plt.title(r'$\alpha$')
        
        plt.subplot(326)
        plt.scatter(x, y, s=s, alpha=.4, c=self.rhoBid, cmap=cmap)
        plt.title(r'$\rho_{bid}$')
        
        fig.suptitle('size = Total final wealth, color = parameter')
        fig.supxlabel('Trading action taken')
        fig.supylabel('Times of resolved trading')

        plt.tight_layout()

        plt.savefig(self.plotPath+'/Trading_attempts', dpi=120)
        plt.close()
       
    def forecast_preference(self, report=False):

        from scipy import stats
    
        #QF = self.RL_variable('QForecasting')
        aF = self.RL_variable('actionForecasting')

        z = aF[:, self.tInit:].mean(axis=1)*100 #should be tInitLearn
 
        cmap = 'coolwarm_r'
        norm = mpl.colors.Normalize(vmin=0,vmax=100)
      
        #y = np.log( self.Atotal[:, -100:].mean(axis=1) )
      
        fig = plt.figure(figsize=(12, 6))
        
        x = self.tau
        y = self.kappa
        plt.subplot(131) 
        plt.scatter(x, y, alpha=.2, c=z, cmap='coolwarm_r')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\kappa$')
        #plt.colorbar()
        #plt.ylabel('Forecasting strategy'+r'$(Q_1-Q_0)$')

        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Fundamentalism (%)', rotation=270)


        plt.subplot(132)    
        y = aF.mean(axis=1)
        plt.scatter(x, y, alpha=.2, c=z, cmap='coolwarm_r')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$a_{\mathcal{F}}$')

        slope, intercept, r, p, se = stats.linregress(x, y)
        print('slope tau vs fund: ', slope, r, p)

        unique_x = np.unique(x)        
        plt.plot(unique_x, np.poly1d(np.polyfit(x, y, 1))(unique_x), color='k', linewidth=2)

        plt.subplot(133)    
        x = self.kappa
        plt.scatter(x, y, alpha=.2, c=z, cmap='coolwarm_r')
        plt.xlabel(r'$\kappa$')
        plt.ylabel(r'$a_{\mathcal{F}}$')
        
        plt.xscale('log')

        slope, intercept, r, p, se = stats.linregress(x, y)
        print('slope kappa vs fund: ', slope, r, p)
        
        slope, intercept, r, p, se = stats.linregress(np.log10(x), y)
        print('slope on log10: ', slope, r, p)
        
        X = np.zeros((np.shape(self.tau)[0], 2))
        X[:, 0] = self.tau
        X[:, 1] = np.log10(self.kappa)

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression

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

        title = f'forecast_preference_new.pdf' if not report else 'tau vs forecast preference'
        self.reporting(fig, title, report=report)

        plt.close()

       
    def forecast_preference2(self, report=False):
    
        QF = self.RL_variable('QForecasting')
        y = (QF[:, -500:, 0, 1]-QF[:, -500:, 0, 0]).mean(axis=1)
        #yFM = (QF[:, -500:, 1]-QF[:, -500:, 2]).mean(axis=1)
        #yTM = (QF[:, -500:, 0]-QF[:, -500:, 2]).mean(axis=1)
      
        s = np.log( self.Atotal[:, -100:].mean(axis=1) )
      
        fig = plt.figure(figsize=(12, 6))
        
        x = self.tau
        plt.subplot(121) 
        plt.scatter(x, y, s=s, alpha=.4, c=self.kappa, cmap='turbo', label='F-T')
        #plt.scatter(x, yFM, s=s, alpha=.15, c=[.8, .5, .5], label='F-M')
        #plt.scatter(x, yTM, s=s, alpha=.15, c=[.5, .5, .8], label='T-M')
        plt.xlabel(r'$\tau$')
        plt.ylabel('Forecasting strategy'+r'$(Q_1-Q_0)$')

        x = self.kappa      
        plt.subplot(122) 
        plt.scatter(x, y, s=s, alpha=.4, c=self.tau, cmap='turbo')
        plt.xlabel(r'$\kappa$')

        y = (QF[:, -500:, 1, 1]-QF[:, -500:, 1, 0]).mean(axis=1)

        #x = self.tau
        #plt.subplot(223) 
        #plt.scatter(x, y, s=s, alpha=.4, c=self.kappa, cmap='turbo', label='F-T')
        #plt.scatter(x, yFM, s=s, alpha=.15, c=[.8, .5, .5], label='F-M')
        #plt.scatter(x, yTM, s=s, alpha=.15, c=[.5, .5, .8], label='T-M')
        #plt.xlabel(r'$\tau$')
        #plt.ylabel('Forecasting strategy'+r'$(Q_1-Q_0)$')

        #x = self.kappa      
        #plt.subplot(224) 
        #plt.scatter(x, y, s=s, alpha=.4, c=self.tau, cmap='turbo')
        #plt.xlabel(r'$\kappa$')

        title = f'forecast_preference.pdf' if not report else 'tau vs forecast preference'
        self.reporting(fig, title, report=report)

        plt.close()
      
        #fig = plt.figure(figsize=(6, 6))
        
        #x = self.tau
        #y2 = self.kappa
        #plt.scatter(x, y2, s=s, alpha=.4, c=y, cmap='turbo')
        #plt.xlabel(r'$\tau$')
        #plt.ylabel(r'$\kappa$')
      
        #title = f'forecast_preference2.pdf' if not report else 'tau vs forecast preference'
        #self.reporting(fig, title, report=report)

        #plt.close()
    
    def success_parameter(self, report=False):
    
        #c = [.2, 0, .2]
        c = range(self.I)

        fig = plt.figure(figsize=(12, 12))
        
        y = np.log10( self.Atotal[:, -1] )
        y0 = np.log10( self.Atotal[0, 0] )
        s = self.VolumeOwned[:, 0, -100:].mean(axis=1) / 20

        x = self.kappa
        plt.subplot(321)
        plt.scatter(x, y, s=s, alpha=.4, c=c, cmap='turbo')
        plt.xlabel(r'$\kappa$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        plt.axhline(y0, color=[.5, .5, .5], linewidth=.5)
        
        x = self.g*self.gSkewness
        plt.subplot(322)
        plt.scatter(x, y, s=s, alpha=.4, c=c, cmap='turbo')
        plt.xlabel(r'$g$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        plt.axhline(y0, color=[.5, .5, .5], linewidth=.5)
        
        x = self.tau
        plt.subplot(323)
        plt.scatter(x, y, s=s, alpha=.4, c=c, cmap='turbo')
        plt.xlabel(r'$\tau$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        plt.axhline(y0, color=[.5, .5, .5], linewidth=.5)
                
        QF = self.RL_variable('QForecasting')
        #x = QF[:, -1, 1]-QF[:, -1, 0]
        plt.subplot(324)
        #plt.scatter(x, y, s=s, alpha=.4, c=c, cmap='turbo')
        plt.xlabel('Forecasting strategy'+r'$(Q_1-Q_0)$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        plt.axhline(y0, color=[.5, .5, .5], linewidth=.5)
        
        try:
            
            x = self.alphaConf-self.alphaDisc
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, c=c, cmap='turbo')
            plt.xlabel(r'$\alpha$'+' bias')
            plt.axhline(y0, color=[.5, .5, .5], linewidth=.5)            
        except:
            
            x = self.alphaConf
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, c=c, cmap='turbo')
            plt.xlabel(r'$\alpha$')
            plt.axhline(y0, color=[.5, .5, .5], linewidth=.5)            
        
        x = self.rhoBid
        plt.subplot(326)
        plt.scatter(x, y, s=s, alpha=.4, c=c, cmap='turbo')
        plt.xlabel(r'$\rho_{bid}$')
        plt.axhline(y0, color=[.5, .5, .5], linewidth=.5)
        
        fig.suptitle('size = mean final Stock Volume owned, color=trader ID')
        fig.supylabel('Total final wealth ($)')

        plt.tight_layout()

        plt.savefig(self.plotPath+'/Success_parameters', dpi=120)

        title = f'Success_parameters.pdf' if not report else 'Agent parameters vs final wealth (training)'
        self.reporting(fig, title, report=report)

        plt.close()
        
        
    def resolved_trading_parameters(self):
    
        c = [.2, 0, .2]

        fig = plt.figure(figsize=(12, 12))
        
        normalise = (abs(self.actionT[:, 0, :-1]-1) > 0).sum(axis=1)
        y = (abs(self.VolumeOwned[:, 0, 1:]-self.VolumeOwned[:, 0, :-1]) > 0).sum(axis=1) / normalise * 100
        s = self.Atotal[:, -1]/2000000

        x = self.kappa
        plt.subplot(321)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$\kappa$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        x = self.g
        plt.subplot(322)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$g$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        x = self.tau
        plt.subplot(323)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$\tau$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        QF = self.RL_variable('QForecasting')
        #x = QF[:, -1, 1]-QF[:, -1, 0]
        plt.subplot(324)
        #plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel('Forecasting strategy')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)

        try:
            
            x = self.alphaConf-self.alphaDisc
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, color=c)
            plt.xlabel(r'$\alpha$'+' bias')
            
        except:
            
            x = self.alphaConf
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, color=c)
            plt.xlabel(r'$\alpha$')
            
        x = self.rhoBid
        plt.subplot(326)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$\rho_{bid}$')

        fig.suptitle('size = Total final wealth')
        fig.supylabel('Times of resolved trading (% of trading actions taken)')

        plt.tight_layout()

        plt.savefig(self.plotPath+'/Resolved_trading_vs_parameters_Percentage', dpi=120)

        plt.close()
        
        # Unnormalised figure
        
        fig = plt.figure(figsize=(12, 12))

        y = (abs(self.VolumeOwned[:, 0, 1:]-self.VolumeOwned[:, 0, :-1]) > 0).sum(axis=1)
        s = self.Atotal[:, -1]/2000000

        x = self.kappa
        plt.subplot(321)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$\kappa$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        x = self.g
        plt.subplot(322)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$g$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        x = self.tau
        plt.subplot(323)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$\tau$')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)
        
        QF = self.RL_variable('QForecasting')
        #x = QF[:, -1, 1]-QF[:, -1, 0]
        plt.subplot(324)
        #plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel('Forecasting strategy')
        #plt.xscale('log', base=2)
        #plt.yscale('log', base=2)

        try:
            
            x = self.alphaConf-self.alphaDisc
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, color=c)
            plt.xlabel(r'$\alpha$'+' bias')
            
        except:
            
            x = self.alphaConf
            plt.subplot(325)
            plt.scatter(x, y, s=s, alpha=.4, color=c)
            plt.xlabel(r'$\alpha$')
            
        x = self.rhoBid
        plt.subplot(326)
        plt.scatter(x, y, s=s, alpha=.4, color=c)
        plt.xlabel(r'$\rho_{bid}$')

        fig.suptitle('size = Total final wealth')
        fig.supylabel('Times of resolved trading')

        plt.tight_layout()

        plt.savefig(self.plotPath+'/Resolved_trading_vs_parameters', dpi=120)

        plt.close()
        
    #%%
    
    def SYMBA_parameter_space(self):
    
        fig = plt.figure(constrained_layout=True, figsize=(10, 7))
        gs1 = fig.add_gridspec(nrows=10, ncols=12)
        
        self.ax = fig.add_subplot(gs1[:, :11], projection='3d')     # top left     (si
        axc = fig.add_subplot(gs1[:, 11:])     # top right     (size: 12x7  - 60x35)

        p = self.ax.scatter(self.kappa, self.g, self.tau,
                color=[.2, 0, .2],
                #c=self.Price.mean(axis=1)-self.FundamentalValues,
                #cmap='jet',
                s = (self.Atotal[:, -1])/100,
                alpha=.4)

        self.ax.set_xlabel(r'$\kappa$')
        self.ax.set_ylabel(r'$g$')
        self.ax.set_zlabel(r'$\tau$')

        axc.set_axis_off()
        #cb = fig.colorbar(p, ax=axc, label=r'$\bar{Price} - FV$' )#r'$g$'+' gesture param.')
        
        plt.savefig(self.plotPath+'/SYMBA_parameter_space', dpi=120)

        plt.close()
        
    def SYMBA_parameter_space_html(self):
    
        import plotly.graph_objects as go

	# Map radius, angle pairs to x, y, z points.
        x, y, z = self.rhoBid, self.g, self.tau
	
        marker = {'size': np.log10(self.Atotal[:, -1]+10)*3, 'opacity':0.5, 'color':self.Atotal[:, -1], 'colorscale':'turbo', 'colorbar':dict(thickness=10)}
        layout = go.Layout(hovermode=False)
        ### PLOT
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
		mode='markers',
		marker=marker,
		line=None,
		)
		], layout=layout)
        
        fig.update_layout(
        #title='something', 
        #autosize=False,
        #width=500, 
        #height=500,
        #margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
        xaxis_title='ρ',
        yaxis_title='g',
        zaxis_title='τ'),)	

        #fig.show()
        #fig.close()

        ### EXPORT TO HTML
        # Please, execute `help(fig.write_html)` to learn about all the
        # available keyword arguments to control the output
        fig.write_html(self.plotPath+"/SYMBA_parameter_space.html", include_plotlyjs=True, full_html=True)
        
    #%%
     
    def create_frames_dynamics(self, giff, a=None):

        skip = 50 #20

        for t in tqdm( range(0, self.T, skip) ):
                            
            if giff == 'QF':
                    
                self.QF_space(t)
                    
            elif giff == 'QT':
                    
                self.QT_space(t)
                
            elif giff == 'trading':
                
                self.trading(t)
                
            elif giff == 'wealth':
                
                self.wealth(t)
                
            self.ax.set_title(t)
            plt.savefig(self.plotPath+self.giffAssemblyFolder+f'/{giff}-{t}.png')
            
            plt.close()
        
        filename = f'/{giff}_dynamics.gif' if a is None else f'/{giff}_dynamics_a0_{a}.gif'
            
        with imageio.get_writer(self.plotPath+filename, mode='i') as writer:
            
            for t in np.arange(0, self.T, skip):
                image = imageio.imread(self.plotPath+self.giffAssemblyFolder+f'/{giff}-{t}.png')
                writer.append_data(image)

        [ os.remove(self.plotPath+self.giffAssemblyFolder+f'/{giff}-{t}.png') for t in np.arange(0, self.T, skip) ]


    def create_frames_rotation(self, giff, a=None):

               #azim=self.ax.get_axes().azim
        
        for angle in tqdm( range(0,360,1) ):

            giffFrameName = f'/{giff}-angle-{angle}.png'
            giffFramePath = self.plotPath+self.giffAssemblyFolder+giffFrameName
            
            if giff == 'QF':

                self.QF_space(-1)
                    
            elif giff == 'QT':

                self.QT_space(-1)
            
            self.ax.view_init(azim=45+angle) #elev=10., 
            #plt.title()
            plt.savefig(giffFramePath)
            plt.close()
        
        with imageio.get_writer(self.plotPath+f'/{giff}_rotation_a0_{a}.gif', mode='i') as writer:
                
            for angle in np.arange(0, 360, 1):
                
                giffFrameName = f'/{giff}-angle-{angle}.png'
                giffFramePath = self.plotPath+self.giffAssemblyFolder+giffFrameName

                image = imageio.imread(giffFramePath)
                writer.append_data(image)
                os.remove(giffFramePath)

        #[ os.remove(giffFramePath) for angle in np.arange(0, 360, 1) ]
    
    #@gif.frame
    def create_giff(self, giff):

        os.mkdir(self.plotPath+self.giffAssemblyFolder) if os.path.isdir(self.plotPath+self.giffAssemblyFolder) == False else None
        
        if giff == 'QF':
            
            ### QF space a0 ###
            self.Q0 = self.QF[:, :, 0]
            self.Q1 = self.QF[:, :, 1]
            self.Q2 = self.QF[:, :, 2]

            print(f'Create giff {giff} dynamics...')
            self.create_frames_dynamics(giff)

            ### Rotation inspection
            print(f'Create rotating giff {giff} terminal step...')
            self.create_frames_rotation(giff)
            
        elif giff == 'QT':
            
            for a in [0, 4]:

                ### QT space a0 = a ###
                self.Q0 = self.QT[:, :, :, :, :, a+0]
                self.Q1 = self.QT[:, :, :, :, :, a+1]
                self.Q2 = self.QT[:, :, :, :, :, a+2]
            
                print(f'Create giff {giff} dynamics... a0 = {a}')
                self.create_frames_dynamics(giff, a)

                ### Rotation inspection
                print(f'Create rotating giff {giff} terminal step... a0 = {a}')
                self.create_frames_rotation(giff, a)
                
        elif giff == 'trading' or giff == 'wealth':
            
            self.create_frames_dynamics(giff)            
            
    def wealth(self, t):
        
        plt.figure(figsize=(10, 5))
        
        xmin = self.Atotal.min(axis=(0,1))*0.95
        xmax = self.Atotal.max(axis=(0,1))*1.05
        
        self.ax = plt.subplot(111)
        self.ax.hist(self.Atotal[:, 0], color='k', alpha=.5, bins=np.logspace(xmin, xmax, 1000), label=r'$t_{0}$', axes=self.ax) #, label='$t_0$')
        self.ax.hist(self.Atotal[:, t], color='r', alpha=.5, bins=np.logspace(xmin, xmax, 1000), axes=self.ax) #, bins=hist[1]) #, label='$t$')
        
        self.ax.set_xscale('log')
            
    def trading(self, t):
        
            buyers = np.where( self.actionT[:, 0, t] < 0)[0]
            sellers = np.where( self.actionT[:, 0, t] > 0)[0]
        
            fig, self.ax = plt.subplots(1, 1, figsize=(10, 5))              
            self.ax.errorbar(self.VolumeOwned[:, 0, t], self.kappa[:, 0], yerr=self.g*self.Price[0, t], fmt='o', color='k') #, markersize=self.Atotal[:, t])
            
            self.ax.errorbar(self.VolumeOwned[buyers, 0, t], self.kappa[buyers, 0], yerr=self.g[buyers]*self.Price[0, t], fmt='o', color='g') #, markersize=self.Atotal[:, t])
            self.ax.errorbar(self.VolumeOwned[sellers, 0, t], self.kappa[sellers, 0], yerr=self.g[sellers]*self.Price[0, t], fmt='o', color='r') #, markersize=self.Atotal[:, t])
            
            plt.axhline(y = self.Price[0, t], color='k', alpha=.5)
            
            self.ax.set_xlim([self.VolumeOwned[:, 0, :].min().min()-10, self.VolumeOwned[:, 0, :].max().max()+10])
            #self.ax.set_xscale('log')
            
            ymin = min(self.FundamentalValues[:, 0])-max(self.g)*min(self.Price[0, :])
            ymax = max(self.FundamentalValues[:, 0])+max(self.g)*max(self.Price[0, :])
            
            self.ax.set_ylim([ymin, ymax])

    def QF_space(self, t):

        fig = plt.figure(constrained_layout=True, figsize=(10, 7))
        gs1 = fig.add_gridspec(nrows=10, ncols=12)
        
        self.ax = fig.add_subplot(gs1[:, :11], projection='3d')     # top left     (si
        axc = fig.add_subplot(gs1[:, 11:])     # top right     (size: 12x7  - 60x35)

        p = self.ax.scatter(self.Q0[:, t],
                self.Q1[:, t],
                self.Q2[:, t],
                c=self.tau,
                cmap='jet',
                #c=rho)
                s = (self.Atotal[:, -1])/100000,
                #s=(VolumeOwned[-1, :, 0]+10)/10,
                alpha=.4)

        #breakpoint()
        self.ax.set_xlim([self.Q0.min(axis=(0, 1)), self.Q0.max(axis=(0, 1))])
        self.ax.set_ylim([ self.Q1.min(axis=(0, 1)), self.Q1.max(axis=(0, 1))])
        self.ax.set_zlim([self.Q2.min(axis=(0, 1))*.9, self.Q2.max(axis=(0, 1))*1.1])
        self.ax.set_xlabel('$Q_0$ \n (MR)')
        self.ax.set_ylabel('$Q_1$ \n (MA)')
        self.ax.set_zlabel('$Q_2$ \n (TF)')

        axc.set_axis_off()
        #cb = fig.colorbar(p, ax=axc, label=r'$\tau$' )#r'$g$'+' gesture param.')
        cb = fig.colorbar(p, ax=axc, label=r'$\tau$' )#r'$g$'+' gesture param.')
        
        
    def QT_space(self, t):

        fig = plt.figure(constrained_layout=True, figsize=(9, 9))
        gs1 = fig.add_gridspec(nrows=10, ncols=12)
        
        self.ax = fig.add_subplot(gs1[0:8, :11], projection='3d')     # top left     (size: 12x7  - 60x35)
        axc = fig.add_subplot(gs1[0:8, 11:])     # top right     (size: 12x7  - 60x35)
        ax2 = fig.add_subplot(gs1[8:, :])    # top right    (size: 12x13 - 60x65)

        x, y, z = np.sum( self.Q0[:, t], axis=(1, 2, 3)), np.sum( self.Q2[:, t], axis=(1, 2, 3)), np.sum( self.Q1[:, t], axis=(1, 2, 3))

        #3D plot
        p = self.ax.scatter(x, y, z,
                    c=self.g, #self.Price.mean(axis=1)-self.FundamentalValues, #self.tau,
                    cmap='jet',
                    #c=rho)
                    s = (self.Atotal[:, -1])/200,
                    #s=(VolumeOwned[-1, :, 0]+10)/10,
                    alpha=.4)

            #breakpoint()
        self.ax.set_xlim([np.sum( self.Q0, axis=(2, 3, 4)).min(axis=(0, 1)), np.sum( self.Q0, axis=(2, 3, 4)).max(axis=(0, 1))])
        self.ax.set_ylim([np.sum( self.Q2, axis=(2, 3, 4)).min(axis=(0, 1))*.9, np.sum( self.Q2, axis=(2, 3, 4)).max(axis=(0, 1))*1.1])
        self.ax.set_zlim([np.sum( self.Q1, axis=(2, 3, 4)).min(axis=(0, 1)), np.sum( self.Q1, axis=(2, 3, 4)).max(axis=(0, 1))])

        #self.ax.set_xlabel('$Q_{a1}$ = 0 \n (Buy)')
        #self.ax.set_ylabel('$Q_{a1}$ = 2 \n (Sell)')
        #self.ax.set_zlabel('$Q_{a1}$ = 1 \n (Keep)')
        
        self.ax.set_xlabel('$Q_{a}$ = 0 \n (-g)')
        self.ax.set_ylabel('$Q_{a}$ = 2 \n (+g)')
        self.ax.set_zlabel('$Q_{a}$ = 1 \n (0)')
        
        #self.ax.set_box_aspect([np.ptp(i) for i in data])  # equal aspect ratio

        #cb = fig.colorbar(p, ax=axc, shrink=.8, label='Total final wealth ($)')
        axc.set_axis_off()
        #cb = fig.colorbar(p, ax=axc, label=r'$\tau$' )#r'$g$'+' gesture param.')
        #cb = fig.colorbar(p, ax=axc, label=r'$\bar{Price} - FV$' )#r'$g$'+' gesture param.')
        cb = fig.colorbar(p, ax=axc, label='$g$' )#r'$g$'+' gesture param.')

        ### Price
        ax2.plot(self.Price[0, :t], color='k')
        ax2.set_xlim([0, np.shape(self.Price)[1]])
        ax2.set_ylim([self.Price[0, :].min()*.95, self.Price[0, :].max()*1.05])
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Stock price')

        #plt.tight_layout()
        
#%%

    def QT_radar_video(self):
        
        self.video = 'QT radar'
        
        statesT = [ [sT0, sT1, sT2] for sT0 in range(4) for sT1 in range(2) for sT2 in range(3)] 
        
        for stateT in statesT:

            self.sT0, self.sT1, self.sT2 = stateT
            
            self.cmap = plt.get_cmap('turbo')
            self.norm = plt.Normalize( np.log10(self.Atotal[:, -1]).min(), np.log10(self.Atotal[:, -1]).max() )
            self.colors = [self.cmap( self.norm( np.log10(self.Atotal[trader, -1]) ) ) for trader in range(self.I)]

            self.set_frame()

    def QT_space_video(self):
        
        self.video = 'QT space'
        
        statesT = [ [sT0, sT1, sT2] for sT0 in range(3) for sT1 in range(2) for sT2 in range(3)] 
        
        for stateT in statesT:

            self.sT0, self.sT1, self.sT2 = stateT

            self.set_frame()
        
    def QT_space_video_plot(self):

        x = self.QT[:, self.t, self.sT0, self.sT1, self.sT2, 0]
        y = self.QT[:, self.t, self.sT0, self.sT1, self.sT2, 2]
        z = self.QT[:, self.t, self.sT0, self.sT1, self.sT2, 1]
        
        #breakpoint()

        #3D plot
        p = self.ax.scatter(x, y, z,
                    c=self.g, #self.Price.mean(axis=1)-self.FundamentalValues, #self.tau,
                    cmap='jet',
                    #c=rho)
                    s = self.Atotal[:, -1]/5000,
                    #s=(VolumeOwned[-1, :, 0]+10)/10,
                    alpha=.4)

            #breakpoint()
        QTx = self.QT[:, :, self.sT0, self.sT1, self.sT2, 0]
        QTy = self.QT[:, :, self.sT0, self.sT1, self.sT2, 2]
        QTz = self.QT[:, :, self.sT0, self.sT1, self.sT2, 1]
        
        self.ax.set_xlim([QTx.min(axis=(0, 1)), QTx.max(axis=(0, 1)) ])
        self.ax.set_ylim([QTy.min(axis=(0, 1)), QTy.max(axis=(0, 1)) ])
        self.ax.set_zlim([QTz.min(axis=(0, 1)), QTz.max(axis=(0, 1)) ])

        #self.ax.set_xlabel('$Q_{a1}$ = 0 \n (Buy)')
        #self.ax.set_ylabel('$Q_{a1}$ = 2 \n (Sell)')
        #self.ax.set_zlabel('$Q_{a1}$ = 1 \n (Keep)')
        
        self.ax.set_xlabel('$Q_{a}$ = 0 \n buy') #(-g)')
        self.ax.set_ylabel('$Q_{a}$ = 2 \n sell') #(+g)')
        self.ax.set_zlabel('$Q_{a}$ = 1 \n hold') #(0)')
        
        #self.ax.set_box_aspect([np.ptp(i) for i in data])  # equal aspect ratio

        #cb = fig.colorbar(p, ax=axc, shrink=.8, label='Total final wealth ($)')
        #self.axc.set_axis_off()
        #cb = fig.colorbar(p, ax=axc, label=r'$\tau$' )#r'$g$'+' gesture param.')
        #cb = fig.colorbar(p, ax=axc, label=r'$\bar{Price} - FV$' )#r'$g$'+' gesture param.')
        #cb = self.fig.colorbar(p, ax=self.axc, label='$g$' )#r'$g$'+' gesture param.')

        ### Price
        self.ax2.plot(self.Price[0, :self.t], color='k')
        self.ax2.set_xlim([0, np.shape(self.Price)[1]])
        self.ax2.set_ylim([self.Price[0, :].min()*.95, self.Price[0, :].max()*1.05])
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Stock price ($)')

    def QF_convergence_learning(self):

        # Titles for the subplots
        titles_top = [r'$h \leq 0 \qquad & \qquad p \geq 0.01$', r'$0 < h \leq 1 \qquad & \qquad p \geq 0.01$', r'$h > 1 \qquad & \qquad p \geq 0.01$']
        titles_bottom = [r'$h \leq 0 \qquad & \qquad p < 0.01$', r'$0 < h \leq 1 \qquad & \qquad p < 0.01$', r'$h > 1 \qquad & \qquad p < 0.01$']

        # Assigning titles efficiently
        titles = titles_top + titles_bottom

        QF = self.RL_variable('QForecasting')

        T = range(self.T)
        traders = np.random.uniform(0, self.I, size=20).astype(int)

        #fig, axs = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)
        fig, axs = plt.subplots(7, 1, figsize=(5, 8), sharex=True)
        axs = axs.flatten()

        # Colors and alpha
        action_menu = {'0': [.65, 0, 0], '1': [0, 0, .65]}
        alpha_value = 0.02
        mean_alpha = 1.0  # Full opacity for mean lines

        # Set common y-limits
        y_min, y_max = QF.min(), QF.max() * 1.05
        
        price = self.Price[0] #(self.Price[0]-self.Price[0, 0]) / max(self.Price[0])

        for axn, ax in enumerate(axs[:-1]):
            # Set the horizontal reference line
            ax.axhline(y=0, color=[.5, .5, .5], lw=0.5)

            # Plot each trader's data with the desired alpha and colors
            for trader in traders:
                for label in action_menu.keys():  # corresponding to colors
                    ax.plot(T, np.squeeze(QF[trader, :, axn, int(label)]), color=action_menu[label], alpha=alpha_value, lw=0.5)

            # Plot the mean lines
            for label in action_menu.keys():
                ax.plot(T, np.mean(QF[:, :, axn, int(label)], axis=0), color=action_menu[label], alpha=mean_alpha, lw=0.85)

            #ax.plot(T, price, lw=0.5, color='k')

            # Set titles, limits, and labels
            ax.set_title(titles[axn], fontsize=10)
            ax.set_xlim([0, self.T])
            ax.set_ylim([y_min, y_max])
            # Set xlabel for the bottom row only
            #ax.set_xlabel(r'Trading day (t)', fontsize=10) if axn >= 5 else None #axn >= 3 else None
            ax.set_ylabel(r'$Q_{\mathcal{F}}$') #if (axn % 3) == 0 else None
            ax.axvline(x=4000, color=[.5, .5, .5], ls='--', lw=0.5)
            ax.text(-0.20, 1.025, string.ascii_lowercase[axn], transform=ax.transAxes, size=12, weight='bold')

        axs[-1].plot(T, price, lw=0.5, color='k', label=r'$P$')
        axs[-1].axvline(x=4000, color=[.5, .5, .5], ls='--', lw=0.5)
        axs[-1].plot(T, self.fundamentalValueTrue[0], lw=0.5, color=[.5, .5, .5], label=r'$f$')
        axs[-1].set_ylabel('Stock price')
        #axs[-1].legend()
        axs[-1].set_xlabel(r'Trading day (t)', fontsize=10) if axn >= 5 else None #axn >= 3 else None
        axs[-1].text(-0.20, 1.025, string.ascii_lowercase[6], transform=axs[-1].transAxes, size=12, weight='bold')
        
        plt.tight_layout()
            
        #plt.legend()
        plt.savefig(self.plotPath+'/convergence_QF_micro_mean.pdf', dpi=120)

        plt.close()

    def QT_convergence_learning(self, prob=True):

        QT = self.RL_variable('QTrading')[:, :, 0]

        stateT = self.RL_variable('stateTrading')[..., 1] #only one level in s0

        T = range(self.T)
        traders = np.random.uniform(0, self.I, size=100).astype(int)
  
        s0Name = ['Single state'] #, 'M']
  
        #s1Name = [r'$V_{-} \, \, & \, \, H \leq \hat{P}$',
        #r'$V_{+} \,\, & \,\, P_{\text{abp}} \leq H \leq \hat{P}$',
        #r'$V_{+} \,\, & \,\, H \leq P_{\text{abp}} < \hat{P}$',
        #r'$V_{+} \,\, & \,\, H \leq \hat{P} \leq P_{\text{abp}}$',
        
        #r'$V_{-} \,\, & \,\, \hat{P} < H$',
        #r'$V_{+} \,\, & \,\, P_{\text{abp}} < \hat{P} < H$',
        #r'$V_{+} \,\, & \,\, \hat{P} \leq P_{\text{abp}} \leq H$',
        #r'$V_{+} \,\, & \,\, \hat{P} < H < P_{\text{abp}} $']

        s1Name = [r'$V_{-} \quad & \quad H \leq \hat{P}$',
        r'$V_{+} \quad & \quad P_{\text{abp}} \leq H \leq \hat{P}$',
        r'$V_{+} \quad & \quad H \leq P_{\text{abp}} < \hat{P}$',
        r'$V_{+} \quad & \quad H \leq \hat{P} \leq P_{\text{abp}}$',
        
        r'$V_{-} \quad & \quad \hat{P} < H$',
        r'$V_{+} \quad & \quad P_{\text{abp}} < \hat{P} < H$',
        r'$V_{+} \quad & \quad \hat{P} \leq P_{\text{abp}} \leq H$',
        r'$V_{+} \quad & \quad \hat{P} < H < P_{\text{abp}} $']
        
        if prob:
            
            for i in range(self.I):
                QT[i] *= self.beta[i]           
            
            QT = np.exp( QT ) 

            QT[..., 0, :2] = QT[..., 0, :2]/QT[..., 0, :2].sum(axis=-1, keepdims=True)
            QT[..., 1:4, :] = QT[..., 1:4, :]/QT[..., 1:4, :].sum(axis=-1, keepdims=True)        
        
            QT[..., 4, :2] = QT[..., 4, :2]/QT[..., 4, :2].sum(axis=-1, keepdims=True)
            QT[..., 5:8, :] = QT[..., 5:8, :]/QT[..., 5:8, :].sum(axis=-1, keepdims=True)  
        
        ymin = QT[:, :, axn].mean(axis=0).min()*1.05 if not prob else 0
        ymax = QT[:, :, axn].mean(axis=0).max()*1.05 if not prob else 1
        
        #color_bid = [0, .5, 0]
        #color_hold = [.2, .2, .2]
        #color_ask = [.5, 0, 0]

        color_bid = "#228833" #[0.40, 0.81, 0.94] #cyan
        color_hold = 'k' #[.5, .5, .5] #[.1, .1, .1]
        color_ask = "#AA3377" #[.5, 0, 0]
        
        for s0n, s0 in enumerate(s0Name):
        
            plt.figure(figsize=(10, 6))
        
            #fig, axs = plt.subplots(2, 4, figsize=(8, 4.8), sharex=True) #, sharey=True)
            fig, axs = plt.subplots(2, 4, figsize=(9, 5.4), sharex=True) #, sharey=True)
            axs = axs.flatten()
                
            for axn, ax in enumerate(axs):
                [ ax.plot(T, np.squeeze( QT[trader, :, axn, 0] ), color=color_bid, alpha=.02) for trader in traders ]
                [ ax.plot(T, np.squeeze( QT[trader, :, axn, 1] ), color=color_hold, alpha=.02) for trader in traders ]
                [ ax.plot(T, np.squeeze( QT[trader, :, axn, 2] ), color=color_ask, alpha=.02) for trader in traders ] if axn in [1, 2, 3, 5, 6, 7] else None

                ax.plot(T, np.squeeze( np.mean(QT[:, :, axn, 0], axis=0) ), color=color_bid, alpha=.7)
                ax.plot(T, np.squeeze( np.mean(QT[:, :, axn, 1], axis=0) ), color=color_hold, alpha=.7)
                ax.plot(T, np.squeeze( np.mean(QT[:, :, axn, 2], axis=0) ), color=color_ask, alpha=.7) if axn in [1, 2, 3, 5, 6, 7] else None
                ax.text(-0.15, 1.06, string.ascii_lowercase[axn], transform=ax.transAxes, size=12, weight='bold')
                #axs[axn].text(-0.25, 1.0, string.ascii_lowercase[axn], transform=axs[axn].transAxes, size=12, weight='bold')
                
                #breakpoint()
                #ax.plot(T, np.log10( (stateT[:, :]==axn).sum(axis=0)), color=[.7, .7, 0], alpha=.7, linewidth=.9)
                                
                ax.set_xlim([0, self.T])
                ax.set_ylim([ymin, ymax])
                #ax.set_ylim([-0.05, 0.05]) #zoom
                ax.set_yticklabels('') if axn % 4 > 0 else None
                ax.set_xlabel('Trading day (t)') if axn // 4 > 0 else None
                ax.set_ylabel('$\pi_{\mathcal{T}}$') if axn % 4 == 0 else None

                ax.set_title(s1Name[axn])
                ax.axvline(x=4000, color=[.5, .5, .5], ls='--', lw=0.5)
        
            plt.tight_layout()
            
            title = f'convergence_QT_micro - {s0}'
            title += ' probability' if prob else '' 
            plt.savefig(self.plotPath+'/'+title+'.pdf', dpi=120)

            plt.close()

    def QT_prob_irrational(self):

        QT = self.RL_variable('QTrading')

        T = range(self.T)
        
        s0Name = ['T', 'F'] #, 'M']
  
        s1Name = [r'$V_{-} \, & \, H \leq P$',
        r'$V_{+} \, & \, P_{abp} \leq H \leq P$',
        r'$V_{+} \, & \, H \leq P_{abp} < P$',
        r'$V_{+} \, & \, H \leq P \leq P_{abp}$',
        
        r'$V_{-} \, & \, P < H$',
        r'$V_{+} \, & \, P_{abp} < P < H$',
        r'$V_{+} \, & \, P \leq P_{abp} \leq H$',
        r'$V_{+} \, & \, P < H < P_{abp} $']
        
        for s0n, s0 in enumerate(s0Name):
        
            plt.figure(figsize=(10, 6))
        
            fig, axs = plt.subplots(2, 4, figsize=(10, 6), sharex=True, sharey=True)
            axs = axs.flatten()
            
            thresholds = [1/2, 1/3, 1/3, 1/3, 1/2, 1/3, 1/3, 1/3]
            actions = [[0], [0, 1], [0, 1], [0, 1], [1], [2], [2], [2]]
            colors = ['g', 'k', 'r']
            a = [2, 3, 3, 3, 2, 3, 3, 3]
                        
            for axn, ax in enumerate(axs):
                a_ = a[axn]
                pi = np.exp(QT[:, :, s0n, axn])/np.exp(QT[:, :, s0n, axn, :a_]).sum(axis=-1, keepdims=True)

                for action in actions[axn]:
                    
                    color = colors[action]
                    ax.plot(T, (pi[:, :, action]>thresholds[axn]).mean(axis=0)*100, color=color)
            
                ax.set_title(s1Name[axn])
                ax.set_ylim([0, 100.05])
        
            plt.tight_layout()
            plt.savefig(self.plotPath+f'/QT_prob_irrational - {s0}.pdf', dpi=120)

            plt.close()

    ### State visit stats

    def compute_frequencies(self, state_array, Nstates):
        norm = 100 / self.I
        frequencies = np.zeros((self.T, Nstates))
        for t in range(self.T):
            #counts = np.bincount(state_array[:, t], minlength=Nstates)
            counts = np.array([(state_array[:, t] == state).sum() for state in range(Nstates)])
            frequencies[t] = counts * norm
        return frequencies

    def calculate_statistics(self, frequencies, state_array):
        mean_freq = frequencies.mean(axis=0)
        std_freq = frequencies.std(axis=0)
        median_freq = np.median(frequencies, axis=0)
        never_visited = np.array([(np.sum(np.all(state_array != state, axis=1)) * (100 / self.I)).round(2) for state in range(frequencies.shape[1])])
        return mean_freq, std_freq, median_freq, never_visited

    def save_to_csv_and_latex(self, df, algorithm, caption, label):

        column_reference = df.columns[0]
        subcolumns = [r'$s_{1}$', r'$s_{2}$']
        # Step 1: Split the first column by '&' and rename the resulting columns

        new_cols = df[column_reference].str.split(r' \\, \\& \\, ', expand=True)
        new_cols[0] = new_cols[0]+'$'
        new_cols[1] = '$'+new_cols[1]        
        # Step 2: Rename the new columns for clarity
        #new_cols.columns = subcolumns

        #breakpoint()        
        # Step 3: Insert the new columns as the first two columns in the DataFrame
        df.insert(0, subcolumns[0], new_cols[0])
        df.insert(1, subcolumns[1], new_cols[1])

        df = df.drop(column_reference, axis=1)
    

        # Save the DataFrame to a CSV file
        csv_file_path = self.plotPath+f'/{algorithm}_state_statistics.csv'
        df.to_csv(csv_file_path, index=False)
        print(f'CSV file saved to {csv_file_path}')

        # Convert the DataFrame to a LaTeX table with caption and label
        latex_table = df.to_latex(
            index=False,
            column_format='|c|c|c|c|c|c|',
            float_format="%.2f",
            escape=False,
            caption=caption,
            label=label
        )
        
        # Add table environment and formatting manually
        latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[h] \n\\centering\n')  
        latex_table = latex_table.replace('\\toprule', f'''\\cmidrule{{1-2}} \\multicolumn{{2}}{{|c|}}{{{column_reference}}} & \\multicolumn{{4}}{{c}}{{}}\\\\ \n\\midrule''')

        # Save the LaTeX table to a file
        latex_file_path = self.plotPath+f'/{algorithm}_state_statistics.tex'
        with open(latex_file_path, 'w') as f:
            f.write(latex_table)
        print(f'LaTeX file saved to {latex_file_path}')

    def generate_statistics(self, state_name_key, state_symbol, titles):
        state_array = self.RL_variable(state_name_key) if state_symbol == 'F' else self.RL_variable(state_name_key)[..., 1]
        state_array = state_array.astype(int)
        Nstates = 6 if state_symbol == 'F' else 8
        algorithm = state_name_key.split('state')[-1]
        frequencies = self.compute_frequencies(state_array, Nstates)
        mean_freq, std_freq, median_freq, never_visited = self.calculate_statistics(frequencies, state_array)
        
        # Create a DataFrame with these statistics
        df = pd.DataFrame({
            f'{algorithm} state $s_{{\mathcal{{{state_symbol}}}}}$': titles,
            'Mean ($\%$)': mean_freq.round(2),
            'S.D. ($\%$)': std_freq.round(2),
            'Median ($\%$)': median_freq.round(2),
            'Never visited ($\%$)': never_visited.round(2),
        })

        caption = f"\\textbf{{{algorithm} state statistics: Mean, Standard Deviation, Median percentage, over time steps, of total agents that visit a given {algorithm} state, and percentage of agents that never visited a given {algorithm} state.}}"        
        label = f"tab:{algorithm}_state_statistics"

        # Save to CSV and LaTeX
        self.save_to_csv_and_latex(df, algorithm, caption, label)

    def forecasting_state_visit(self):
        titles = [r'$h \leq 0 \, \& \, p \geq 0.01$', r'$0 < h \leq 1 \, \& \, p \geq 0.01$', r'$h > 1 \, \& \, p \geq 0.01$',
                  r'$h \leq 0 \, \& \, p < 0.01$', r'$0 < h \leq 1 \, \& \, p < 0.01$', r'$h > 1 \, \& \, p < 0.01$']
        self.generate_statistics('stateForecasting', 'F', titles)

    def trading_state_visit(self):
        titles = [r'$V_{-} \, \& \, H \leq \hat{P}$', r'$V_{+} \, \& \, P_{\text{abp}} \leq H \leq \hat{P}$',
                  r'$V_{+} \, \& \, H \leq P_{\text{abp}} < \hat{P}$', r'$V_{+} \, \& \, H \leq \hat{P} \leq P_{\text{abp}}$',
                  r'$V_{-} \, \& \, \hat{P} < H$', r'$V_{+} \, \& \, P_{\text{abp}} < \hat{P} < H$',
                  r'$V_{+} \, \& \, \hat{P} \leq P_{\text{abp}} \leq H$', r'$V_{+} \, \& \, \hat{P} < H < P_{\text{abp}}$']
        self.generate_statistics('stateTrading', 'T', titles)

    def forecasting_state_visit3(self):
        
        algorithm = 'Forecasting'
        state_array = self.RL_variable('stateForecasting').astype(int)

        # Titles for the subplots
        titles_top = [r'$h \leq 0 \, \& \, p \geq 0.01$', r'$0 < h \leq 1 \, \& \, p \geq 0.01$', r'$h > 1 \, \& \, p \geq 0.01$']
        titles_bottom = [r'$h \leq 0 \, \& \, p < 0.01$', r'$0 < h \leq 1 \, \& \, p < 0.01$', r'$h > 1 \, \& \, p < 0.01$']

        # Assigning titles efficiently
        s1Name = titles_top + titles_bottom
     
        Nstates = len(s1Name)
        norm = 100 / self.I
        frequencies = np.zeros((self.T, Nstates))
    
        # Calculate frequencies
        for t in range(self.T):
            counts = np.array([(state_array[:, t] == state).sum() for state in range(Nstates)])
            frequencies[t] = counts *norm
    
        # Calculate mean, standard deviation (S.D.), and median
        mean_freq = frequencies.mean(axis=0)
        std_freq = frequencies.std(axis=0)
        median_freq = np.median(frequencies, axis=0)
        never_visited = np.array( [ (np.sum(np.all(state_array != state, axis=1))*norm ).round(2) for state in range(Nstates) ] )
    
        # Create a DataFrame with these statistics
        df = pd.DataFrame({
            f"{algorithm} state "+"$s_{\mathcal{F}}$": s1Name,
            'Mean ($\%$)': mean_freq.round(2),
            'S.D. ($\%$)': std_freq.round(2),
            'Median ($\%$)': median_freq.round(2),
            'Never visited ($\%$)': never_visited.round(2),
            })
    
        # Save the DataFrame to a CSV file
        csv_file_path = self.plotPath+f'/{algorithm}_state_statistics.csv'
        df.to_csv(csv_file_path, index=False)

        print(f'CSV file saved to {csv_file_path}')

        # Convert the DataFrame to a LaTeX table with caption and label
        latex_table = df.to_latex(index=False, column_format='|l|c|c|c|c|', float_format="%.2f", escape=False, 
                          caption="\\textbf{state statistics: Mean, Standard Deviation, Median percentage of total agents that visit a given state, and percentage of agents that never visited a given state.}", 
                          label=f"tab:{algorithm}_state_statistics")

        # Add table environment and formatting manually
        latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[h] \n\\centering\n')


        # Save the LaTeX table to a file
        with open(self.plotPath+f'/{algorithm}_state_statistics.tex', 'w') as f:
            f.write(latex_table)

    def forecasting_state_visit2(self):
    
        states = self.RL_variable('stateForecasting').astype(int)

        # Titles for the subplots
        titles_top = [r'$h \leq 0 \, & \, p \geq 0.01$', r'$0 < h \leq 1 \, & \, p \geq 0.01$', r'$h > 1 \, & \, p \geq 0.01$']
        titles_bottom = [r'$h \leq 0 \, & \, p < 0.01$', r'$0 < h \leq 1 \, & \, p < 0.01$', r'$h > 1 \, & \, p < 0.01$']

        # Assigning titles efficiently
        titles = titles_top + titles_bottom
        
        unique, counts = np.unique(states, return_counts=True)
        frequencies = dict(zip(titles, counts/counts.sum()*100))
        print('Forecasting state frequencies', frequencies)
        
        never_visited = {state: np.sum(np.all(states != state, axis=1)) for state in range(len(titles))}
        print('Never visited', never_visited)

    def trading_state_visit3(self):
    
        stateTrading = self.RL_variable('stateTrading')[..., 1].astype(int)

        s1Name = [r'$V_{-} \, \& \, H \leq \hat{P}$',
                      r'$V_{+} \, \& \, P_{\text{abp}} \leq H \leq \hat{P}$',
                      r'$V_{+} \, \& \, H \leq P_{\text{abp}} < \hat{P}$',
                      r'$V_{+} \, \& \, H \leq \hat{P} \leq P_{\text{abp}}$',
                      r'$V_{-} \, \& \, \hat{P} < H$',
                      r'$V_{+} \, \& \, P_{\text{abp}} < \hat{P} < H$',
                      r'$V_{+} \, \& \, \hat{P} \leq P_{\text{abp}} \leq H$',
                      r'$V_{+} \, \& \, \hat{P} < H < P_{\text{abp}}$']
     
        Nstates = len(s1Name)
        norm = 100 / self.I
        frequencies = np.zeros((self.T, Nstates))
    
        # Calculate frequencies
        for t in range(self.T):
            counts = np.array([(stateTrading[:, t] == state).sum() for state in range(Nstates)])
            frequencies[t] = counts *norm
    
        # Calculate mean, standard deviation (S.D.), and median
        mean_freq = frequencies.mean(axis=0)
        std_freq = frequencies.std(axis=0)
        median_freq = np.median(frequencies, axis=0)
        never_visited = np.array( [ (np.sum(np.all(stateTrading != state, axis=1))*norm ).round(2) for state in range(Nstates) ] )
    
        # Create a DataFrame with these statistics
        df = pd.DataFrame({
            'Trading state $s_{\mathcal{T}}$': s1Name,
            'Mean ($\%$)': mean_freq.round(2),
            'S.D. ($\%$)': std_freq.round(2),
            'Median ($\%$)': median_freq.round(2),
            'Never visited ($\%$)': never_visited.round(2),
            })
    
        # Save the DataFrame to a CSV file
        csv_file_path = self.plotPath+'/trading_state_statistics.csv'
        df.to_csv(csv_file_path, index=False)

        print(f'CSV file saved to {csv_file_path}')

        # Convert the DataFrame to a LaTeX table with caption and label
        latex_table = df.to_latex(index=False, column_format='|l|c|c|c|c|', float_format="%.2f", escape=False, 
                          caption="\\textbf{Trading State Statistics: Mean, Standard Deviation, Median percentage of total agents that visit a given trading state, and percentage of agents that never visited a given trading state.}", 
                          label="tab:trading_state_statistics")

        # Add table environment and formatting manually
        latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[h] \n\\centering\n')


        # Save the LaTeX table to a file
        with open(self.plotPath+'/trading_state_statistics.tex', 'w') as f:
            f.write(latex_table)

    def trading_state_visit2(self):
    
        stateTrading = self.RL_variable('stateTrading')[..., 1].astype(int)

        s1Name = [r'$V_{-} \, \& \, H \leq \hat{P}$',
        r'$V_{+} \, \& \, P_{\text{abp}} \leq H \leq \hat{P}$',
        r'$V_{+} \, \& \, H \leq P_{\text{abp}} < \hat{P}$',
        r'$V_{+} \, \& \, H \leq \hat{P} \leq P_{\text{abp}}$',
        
        r'$V_{-} \, \& \, \hat{P} < H$',
        r'$V_{+} \, \& \, P_{\text{abp}} < \hat{P} < H$',
        r'$V_{+} \, \& \, \hat{P} \leq P_{\text{abp}} \leq H$',
        r'$V_{+} \, \& \, \hat{P} < H < P_{\text{abp}} $']
        
        frequencies = np.zeros((self.T, len(s1Name)))
        
        for t in range(self.T):
            counts = np.array( [ (stateTrading[:, t] == state).sum() for state in range(len(s1Name)) ] )
            frequencies[t] = counts/self.I*100
            
        print('Trading state frequencies', frequencies.mean(axis=0), frequencies.std(axis=0))
        print(np.median(frequencies, axis=0))

        
        norm = 100/np.shape(stateTrading)[0]
        never_visited = {state: ( np.sum(np.all(stateTrading != state, axis=1))*norm ).round(2) for state in range(8)}
        print('Never visited', never_visited)

    def pricing_state_visit(self):
     
        print('Warning! Agents that do not trade keep a zero in their pricing state matrix!')
    
        states = self.RL_variable('statePricing').astype(int)

        # Titles for the subplots
        titles = [r'$\text{Bid} \& \hat{P} \geq P_{\text{best}}$',
                  r'$\text{Bid} \& \hat{P} < P_{\text{best}}$',
                  r'$\text{Ask} \& \hat{P} \geq P_{\text{best}}$',
                  r'$\text{Ask} \& \hat{P} < P_{\text{best}}$',]

        # Assigning titles efficiently
        titles = titles_top + titles_bottom
        
        unique, counts = np.unique(states, return_counts=True)
        frequencies = dict(zip(titles, counts/counts.sum()*100))
        print('Forecasting state frequencies', frequencies)
        
        never_visited = {state: np.sum(np.all(states != state, axis=1)) for state in range(len(titles))}
        print('Never visited', never_visited)

    def forecasting_transition_matrix(self):
    
        stateTrading = self.RL_variable('stateForecasting').astype(int)

        s1Name = [r'$V_{-} \, & \, H \leq \hat{P}$',
        r'$V_{+} \, & \, P_{\text{abp}} \leq H \leq \hat{P}$',
        r'$V_{+} \, & \, H \leq P_{\text{abp}} < \hat{P}$',
        r'$V_{+} \, & \, H \leq \hat{P} \leq P_{\text{abp}}$',
        
        r'$V_{-} \, & \, \hat{P} < H$',
        r'$V_{+} \, & \, P_{\text{abp}} < \hat{P} < H$']
                
        T = range(252, self.T)

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
                
        ax.set_xticklabels(s1Name, rotation=30)
        ax.set_yticklabels(s1Name, rotation=30)

        # Adjust layout
        plt.tight_layout()

        # Save the plot as a PDF file
        plt.savefig(self.plotPath+'/transition_matrix_forecasting.pdf', format='pdf', dpi=300)

    def trading_transition_matrix(self):
    
        stateTrading = self.RL_variable('stateTrading')[..., 1].astype(int)

        s1Name = [r'$V_{-} \, & \, H \leq \hat{P}$',
        r'$V_{+} \, & \, P_{\text{abp}} \leq H \leq \hat{P}$',
        r'$V_{+} \, & \, H \leq P_{\text{abp}} < \hat{P}$',
        r'$V_{+} \, & \, H \leq \hat{P} \leq P_{\text{abp}}$',
        
        r'$V_{-} \, & \, \hat{P} < H$',
        r'$V_{+} \, & \, P_{\text{abp}} < \hat{P} < H$',
        r'$V_{+} \, & \, \hat{P} \leq P_{\text{abp}} \leq H$',
        r'$V_{+} \, & \, \hat{P} < H < P_{\text{abp}} $']
                
        T = range(252, self.T)

        # Initialize the transition matrix
        num_states = 8
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
        ax.set_xlabel(r'$s_{\mathcal{T}}(t+1)$')
        ax.set_ylabel(r'$s_{\mathcal{T}}(t)$')
        #ax.set_title(r'$s_{\mathcal{T}} \text{Transition Matrix}$')

        # Add a label to the color bar
        cbar = ax.collections[0].colorbar
        cbar.set_label(r'$\pi_{s_{t} \rightarrow s_{t+1}}$', fontsize=14)

        # Set ticks and labels for better readability
        ax.set_xticks(np.arange(num_states) + 0.5)
        ax.set_yticks(np.arange(num_states) + 0.5)
                
        ax.set_xticklabels(s1Name, rotation=30)
        ax.set_yticklabels(s1Name, rotation=30)

        # Adjust layout
        plt.tight_layout()

        # Save the plot as a PDF file
        plt.savefig(self.plotPath+'/transition_matrix.pdf', format='pdf', dpi=300)

    def QT_visiting(self):

        self.stateTrading = self.RL_variable('stateTrading')

        T = range(self.T)
        traders = np.random.uniform(0, self.I, size=100).astype(int)
  
        s0Name = ['T', 'F'] #, 'M']
  
        s1Name = [r'$V_{-} \, & \, H \leq P$',
        r'$V_{+} \, & \, P_{abp} \leq H \leq P$',
        r'$V_{+} \, & \, H \leq P_{abp} < P$',
        r'$V_{+} \, & \, H \leq P \leq P_{abp}$',
        
        r'$V_{-} \, & \, P < H$',
        r'$V_{+} \, & \, P_{abp} < P < H$',
        r'$V_{+} \, & \, P \leq P_{abp} \leq H$',
        r'$V_{+} \, & \, P < H < P_{abp} $']
        
        
        colors = ['k', 'r', 'r', 'r', 'g', [0, .5, .5], [0, .2, .2],  [0, 0, .5]]
        ls = ['-', '--']
        
        plt.figure(figsize=(10, 6))
                                
        for s0n, s0 in enumerate(s0Name):
            
            s0cond = (self.stateTrading[:, :, 0]==s0n)

            for s1 in [0, 1, 4, 5, 6, 7]:
                
                s1cond = (self.stateTrading[:, :, 1]==s1)
                
                plt.plot(T, ( s0cond * s1cond ).sum(axis=0)/self.I*100, color=colors[s1], ls=ls[s0n], label=s1Name[s1]+' // '+s0, lw=1, alpha=.7)
            
        plt.ylim([-0.05, 100.05])    
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plotPath+f'/QT_visiting.pdf', dpi=120)

        plt.close()

    def QP_convergence_learning(self):

        QT = self.RL_variable('QPricing')
        VT = self.RL_variable('VPricing')

        T = range(self.T)
        
        traders = np.array(range(self.I))
        
        tradersgminus = traders[self.gSkewness < 0]
        tradersgminus_subsample = np.random.choice(tradersgminus, size=100)
  
        tradersgplus = traders[self.gSkewness > 0]
        tradersgplus_subsample = np.random.choice(tradersgplus, size=100)
          
        s0Name = ['Bid', 'Ask'] #, 'M']

        s1Name = [r'$\hat{P} \geq P_{\text{best}}$', #r'$\hat{P}_t < P_{\text{auction}, \, t-1^{+} }$'
                  r'$\hat{P} < P_{\text{best}}$'] #r'$\hat{P}_t \geq P_{\text{auction}, \, t-1^{+} }$'
        
        actionName = [[ #gSkew + 1
                       [r'$P_{\text{best bid}}(1-\tilde{g})$', #s = -1 (bid), gskew = +1 (upper row)
                       r'$P_{\text{best bid}}(1+\frac{\tilde{g}}{2})$', #s = -1 (bid), gskew = +1 (upper row), -/2 deflection
                       r'$\hat{P}(1-\tilde{g})$', 
                       r'$\hat{P}(1+\frac{\tilde{g}}{2})$',
                       r'$P_{\text{market}}$'
                       ],
                       [r'$P_{\text{best ask}}(1+\tilde{g})$', #s = 1 (ask), gskew = +1 (upper row)
                       r'$P_{\text{best ask}}(1-\frac{\tilde{g}}{2})$', #s = 1 (ask), gskew = +1 (upper row), -/2 deflection
                       r'$\hat{P}(1+\tilde{g})$', 
                       r'$\hat{P}(1-\frac{\tilde{g}}{2})$',
                       r'$P_{\text{market}}$'
                       ]
                       ],
                       [ #gSkew - 1
                       [r'$P_{\text{best bid}}(1+\tilde{g})$', #s = -1 (bid), gskew = -1 (lower row)
                       r'$P_{\text{best bid}}(1-\frac{\tilde{g}}{2})$', #s = -1 (bid), gskew = -1 (lower row), -/2 deflection
                       r'$\hat{P}(1+\tilde{g})$', 
                       r'$\hat{P}(1-\frac{\tilde{g}}{2})$',
                       r'$P_{\text{market}}$'
                       ],
                       [r'$P_{\text{best ask}}(1-\tilde{g})$', #s = +1 (ask), gskew = -1 (upper row)
                       r'$P_{\text{best ask}}(1+\frac{\tilde{g}}{2})$', #s = +1 (ask), gskew = -1 (upper row), -/2 deflection
                       r'$\hat{P}(1-\tilde{g})$', 
                       r'$\hat{P}(1+\frac{\tilde{g}}{2})$',
                       r'$P_{\text{market}}$'
                       ]
                       ]
                       ]
 
        #self.LimitPrices[traders, stock, 0] = self.LastAuctionPriceReference[traders, stock]
        #self.LimitPrices[traders, stock, 1] = self.LastAuctionPriceReference[traders, stock]
        
        #self.LimitPrices[traders, stock, 2] = self.NextPriceReference[traders, stock]
        #self.LimitPrices[traders, stock, 3] = self.NextPriceReference[traders, stock]
  
        #self.PriceDeflection[traders, stock, 0] = deflection.round(2)
        #self.PriceDeflection[traders, stock, 1] = -(deflection/2).round(2)

        #g = self.gTilde[traders]*self.gSkewness[traders]*sDummy
        #deflection = g * self.NextPriceReference[traders, stock]

        #self.PriceDeflection[traders, stock, 2] = deflection.round(2)
        #self.PriceDeflection[traders, stock, 3] = -(deflection/2).round(2)
        
        #colors = [[
        #          [ #gSkew + 1, bid
        #          [1, 0, 0], #light red
        #          [.5, 0, 0], #dark red
        #          [0, 0, 1], #light blue
        #          [0, 0, .5], #dark blue
        #          [0, 0, 0]
        #          ],
        #          [ #gSkew + 1, ask
        #          [.5, 0, 0], #dark red                  
        #          [1, 0, 0], #light red
        #          [0, 0, .5], #dark blue
        #          [0, 0, 1], #light blue
        #          [0, 0, 0]
        #          ]
        #          ],
        #          [
        #          [ #gSkew -1, bid
        #          [.5, 0, 0], #dark red                  
        #          [1, 0, 0], #light red
        #          [0, 0, .5], #dark blue
        #          [0, 0, 1], #light blue
        #          [0, 0, 0]
        #          ],
        #          [ #gSkew - 1, ask
        #          [1, 0, 0], #light red
        #          [.5, 0, 0], #dark red
        #          [0, 0, 1], #light blue
        #          [0, 0, .5], #dark blue
        #          [0, 0, 0]
        #          ],
        #          ]
        #          ] #black

        colors = [[
                  [ #gSkew + 1, bid
                  [1, 0, 0], #light red
                  [.5, 0, 0], #dark red
                  [0, 0, 1], #light blue
                  [0, 0, .5], #dark blue
                  [0, 0, 0]
                  ],
                  [ #gSkew + 1, ask
                  [1, 0, 0], #light red
                  [.5, 0, 0], #dark red
                  [0, 0, 1], #light blue
                  [0, 0, .5], #dark blue
                  [0, 0, 0]
                  ]
                  ],
                  [
                  [ #gSkew -1, bid
                  [1, 0, 0], #light red
                  [.5, 0, 0], #dark red
                  [0, 0, 1], #light blue
                  [0, 0, .5], #dark blue
                  [0, 0, 0]
                  ],
                  [ #gSkew - 1, ask
                  [1, 0, 0], #light red
                  [.5, 0, 0], #dark red
                  [0, 0, 1], #light blue
                  [0, 0, .5], #dark blue
                  [0, 0, 0]
                  ],
                  ]
                  ] #black

        ymin = -2.5 #QT.min()/2
        ymax = 2.5 #QT.max()/2

        fig, axs = plt.subplots(2, 4, figsize=(8.5, 7), sharex=True) #, sharey=True)        
        #fig, axs = plt.subplots(2, 4, figsize=(11.5, 7.5), sharex=True) #, sharey=True)
        axs = axs.flatten()

        axs[0].text(-0.25, 1.0, string.ascii_lowercase[0], transform=axs[0].transAxes, size=12, weight='bold')
        axs[4].text(-0.25, 1.0, string.ascii_lowercase[1], transform=axs[4].transAxes, size=12, weight='bold')
                
        for s0n, s0 in enumerate(s0Name):
            
            for s1n, s1 in enumerate(s1Name):
            
                axn = s0n * len(s1Name) + s1n
                ax = axs[axn]
                    
                [ [ ax.plot(T, np.squeeze( QT[trader, :, s0n, s1n, aPn] ), color=colors[0][s0n][aPn], alpha=.02) for trader in tradersgplus_subsample ] for aPn, aP in enumerate(actionName[0][s0n]) ] #traders  
                
                #breakpoint()              
                [ ax.plot(T, np.squeeze( np.mean(QT[tradersgplus, :, s0n, s1n, aPn], axis=0) ), color=colors[0][s0n][aPn], alpha=.7, label=aP) for aPn, aP in enumerate(actionName[0][s0n]) ] #mean

                #ax.plot(T, np.squeeze( VT[tradersgplus, :, s0n, s1n].mean(axis=0) ), color='g', alpha=.7) # state value
                
                ax.set_yticklabels('') if axn > 0 else None
                ax.set_ylabel(r'$Q_{\mathcal{P}}$') if axn == 0 else None
                ax.set_xlim([0, self.T])
                ax.set_ylim([-2.1, 3])
                ax.set_title(r'$\text{s0Name[s0n]} \quad & \quad \text{s1Name[s1n]}$')
                ax.axvline(x=4000, color=[.5, .5, .5], ls='--', lw=0.5)
                
                ax.legend(loc=1)
                        
        for s0n, s0 in enumerate(s0Name):
            
            for s1n, s1 in enumerate(s1Name):
            
                axn = s0n * len(s1Name) + s1n + 4
                ax = axs[axn]
                    
                [ [ ax.plot(T, np.squeeze( QT[trader, :, s0n, s1n, aPn] ), color=colors[1][s0n][aPn], alpha=.02) for trader in tradersgminus_subsample ] for aPn, aP in enumerate(actionName[1][s0n]) ] #traders                
                [ ax.plot(T, np.squeeze( np.mean(QT[tradersgminus, :, s0n, s1n, aPn], axis=0) ), color=colors[1][s0n][aPn], alpha=.7, label=aP) for aPn, aP in enumerate(actionName[1][s0n]) ] #mean

                #ax.plot(T, np.squeeze( VT[tradersgminus, :, s0n, s1n].mean(axis=0) ), color='g', alpha=.7) # state value
                
                ax.set_yticklabels('') if axn > 4 else None
                ax.set_ylabel(r'$Q_{\mathcal{P}}$')  if axn == 4 else None
                ax.set_xlim([0, self.T])
                ax.set_ylim([ymin, 1.5])
                ax.set_xlabel('Trading day (t)')
                #ax.set_title(s0Name[s0n]+' & '+s1Name[s1n])
                ax.axvline(x=4000, color=[.5, .5, .5], ls='--')
                
                ax.legend(loc=4)
        
        # Set xlabel for the entire figure
        #fig.text(0.5, 0.04, 'Trading day (t)', ha='center')
        #plt.tight_layout(rect=[0, 0.04, 1, 1])  # Adjust layout to make space for xlabel

        plt.tight_layout()  # Adjust layout to make space for xlabel

        plt.savefig(self.plotPath+f'/convergence_QP_micro.pdf', bbox_inches='tight')

        plt.close()

    def macro_convergence(self, report=False):

        sliding_window_view = np.lib.stride_tricks.sliding_window_view
           
        for j in range(self.J):

            SW = sliding_window_view(self.r[j, :], window_shape = self.Ty)

            fig, axs = plt.subplots(5, 1, figsize=(10, 6), sharex=True, sharey=False)
            axs = axs.ravel()

            ax0 = axs[0]
            x = SW.mean(axis=1)
            ax0.plot(x, color='k')
            ax0.axhline(y=0, color='k', alpha=0.1)
            ax0.set_ylabel('Mean '+r'($\mu[r]$)')
            #ax4.set_yscale('log')
            ax0.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

            ax1 = axs[1]
            x = SW.var(axis=1)
            ax1.plot(x, color='k')
            ax1.set_ylabel('Volatility ( Var[r] )')
            #ax4.set_yscale('log')
            ax1.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

            ax2 = axs[2]
            x = scipy.stats.skew(SW, axis=1)
            ax2.plot(x, color='k')
            ax2.axhline(y=0, color='k', alpha=0.1)
            ax2.set_ylabel('Skew[r]')
            #ax3.set_yscale('log')
            ax2.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
            
            ax3 = axs[3]
            x = scipy.stats.kurtosis(SW, axis=1)
            ax3.plot(x, color='k')
            ax3.axhline(y=0, color='k', alpha=0.1)
            ax3.set_ylabel('Kurtosis[r]')
            #ax4.set_yscale('log')
            ax3.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

            ax4 = axs[4]
            x = [ sm.tsa.acf(SW[t_], nlags=1)[-1] for t_ in range( np.shape(SW)[0] ) ]
            #breakpoint()
            ax4.plot(x, color='k')
            ax4.axhline(y=0, color='k', alpha=0.1)
            #ax1.set_xlabel('Time step (trading days)')
            ax4.set_ylabel(r'$AC_{t=1}[r]$')
            #ax3.set_yscale('log')
            ax4.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

            ax4.set_xlabel('Time step (trading days)')            

            plt.tight_layout()

            title = f'macro_convergence{j}_training.pdf' if not report else 'Market dynamics (training)'
            self.reporting(fig, title, report=report)
            
        return fig

    def convergence_QF(self):
        
        self.dQF = self.QF[:, 1:].sum(axis=2)-self.QF[:, :-1].sum(axis=2)
        self.video = 'learning convergence'
        
        self.set_frame()
        
    def convergence_QF_plot(self):
    
        self.ax.scatter(np.ones(self.I)*self.t, [self.dQF[:, self.t]])
        self.ax.set_xlim([0, self.T])
        self.ax.set_ylim([self.dQF.min(), self.dQF.max()*1.05])
        
    def convergence_QT(self):
        
        self.dQT = self.QT[:, 1:].sum(axis=2)-self.QT[:, :-1].sum(axis=2)
        self.video = 'QT learning convergence'
        
        self.set_frame()
        
    def convergence_QF_plot(self):
    
        self.ax.scatter(np.ones(self.I)*self.t, [self.dQF[:, self.t]])
        self.ax.set_xlim([0, self.T])
        self.ax.set_ylim([self.dQF.min(), self.dQF.max()*1.05])   

    def pareto(self):
                
        self.video = 'pareto'
        
        self.xmax = self.Atotal.max(axis=(0,1))*1.05
        self.xmaxEquity = self.Aequity.max(axis=(0,1))*1.05
        self.xmaxAM = self.Wallet.max(axis=(0,1))*1.05
        
        self.set_frame()
        
    def pareto_plot(self):
        
        #breakpoint()
        self.ax1.axhline( y = self.Atotal[:, 0].mean(), color='r', label=r'$t_{0}$') #, label='$t_0$')
        self.ax1.bar( range(self.I), np.sort( self.Atotal[:, self.t])[::-1], color='k', alpha=.5) #, bins=hist[1]) #, label='$t$')
        
        #self.ax.set_xscale('log')
        self.ax1.set_ylim([0, self.xmax])
        self.ax1.yaxis.major.formatter.set_powerlimits((np.log10(1), np.log10(self.xmax)))
        self.ax1.set_ylabel('Total Wealth ($)')
        
        #self.ax1.set_xlim([0, self.I/2])
        self.ax1.set_xlabel('Trader ranking')
        
        self.ax1.legend()
        
        ####
        
        self.ax2.axhline( y = self.Aequity[:, 0].mean(), color='r', label=r'$t_{0}$') #, label='$t_0$')
        self.ax2.bar( range(self.I), np.sort( self.Aequity[:, self.t])[::-1], color='k', alpha=.5) #, bins=hist[1]) #, label='$t$')
        
        #self.ax.set_xscale('log')
        self.ax2.set_ylim([0, self.xmaxEquity])
        self.ax2.yaxis.major.formatter.set_powerlimits((np.log10(1), np.log10(self.xmaxEquity)))
        self.ax2.set_ylabel('Stock value owned ($)')
        
        #self.ax2.set_xlim([0, self.I/2])
        self.ax2.set_xlabel('Trader ranking')
        
        ####
        
        self.ax3.axhline( y = self.Wallet[:, 0].mean(), color='r', label=r'$t_{0}$') #, label='$t_0$')
        self.ax3.bar( range(self.I), np.sort( self.Wallet[:, self.t])[::-1], color='k', alpha=.5) #, bins=hist[1]) #, label='$t$')
        
        #self.ax.set_xscale('log')
        self.ax3.set_ylim([0, self.xmaxAM])
        self.ax3.yaxis.major.formatter.set_powerlimits((np.log10(1), np.log10(self.xmaxAM)))
        self.ax3.set_ylabel('Wallet ($)')
        
        #self.ax3.set_xlim([0, self.I/2])
        self.ax3.set_xlabel('Trader ranking')
        
        plt.tight_layout()
        
    def wealth_video(self):
                
        self.video = 'wealth'
                
        self.xmin = self.Atotal.min(axis=(0,1))*0.95
        self.xmax = self.Atotal.max(axis=(0,1))*1.05
        
        self.xminEquity = self.Aequity.min(axis=(0,1))*0.95
        self.xmaxEquity = self.Aequity.max(axis=(0,1))*1.05
        
        self.xminAM = self.Wallet.min(axis=(0,1))*0.95
        self.xmaxAM = self.Wallet.max(axis=(0,1))*1.05
        
        self.set_frame()
        
    def wealth_video_plot(self):
        
        #breakpoint()
        self.ax1.hist(self.Atotal[:, 0], color='k', alpha=.5, bins=np.linspace(self.xmin, self.xmax, 100), label=r'$t_{0}$') #, label='$t_0$')
        self.ax1.hist(self.Atotal[:, self.t], color='r', alpha=.5, bins=np.linspace(self.xmin, self.xmax, 100)) #, bins=hist[1]) #, label='$t$')
        
        #self.ax.set_xscale('log')
        self.ax1.xaxis.major.formatter.set_powerlimits((np.log10(self.xmin),np.log10(self.xmax)))
        self.ax1.set_xlabel('Total Wealth ($)')
        
        self.ax1.set_ylim([0, self.I/2])
        self.ax1.set_ylabel('Trader counts')
        
        ####
        
        self.ax2.hist(self.Aequity[:, 0], color='k', alpha=.5, bins=np.linspace(self.xminEquity, self.xmaxEquity, 100), label=r'$t_{0}$') #, label='$t_0$')
        self.ax2.hist(self.Aequity[:, self.t], color='r', alpha=.5, bins=np.linspace(self.xminEquity, self.xmaxEquity, 100)) #, bins=hist[1]) #, label='$t$')
        
        #self.ax.set_xscale('log')
        self.ax2.xaxis.major.formatter.set_powerlimits((np.log10(1),np.log10(self.xmaxEquity)))
        self.ax2.set_xlabel('Stock value owned ($)')
        
        self.ax2.set_ylim([0, self.I/2])
        self.ax2.set_ylabel('Trader counts')
        
        ####
        
        self.ax3.hist(self.Wallet[:, 0], color='k', alpha=.5, bins=np.linspace(self.xminAM, self.xmaxAM, 100), label=r'$t_{0}$') #, label='$t_0$')
        self.ax3.hist(self.Wallet[:, self.t], color='r', alpha=.5, bins=np.linspace(self.xminAM, self.xmaxAM, 100)) #, bins=hist[1]) #, label='$t$')
        
        #self.ax.set_xscale('log')
        self.ax3.xaxis.major.formatter.set_powerlimits((np.log10(1),np.log10(self.xmaxAM)))
        self.ax3.set_xlabel('Wallet ($)')
        
        self.ax3.set_ylim([0, self.I/2])
        self.ax3.set_ylabel('Trader counts')
        
        plt.tight_layout()
        
    def scatter_wealth_video(self):

        self.richBorder = self.Aequity[0, 0]+self.Wallet[0, 0]
        self.x = np.linspace(0.001, self.richBorder, 1000)[::-1]
        self.y = [self.richBorder-x_ for x_ in self.x]    
        self.video = 'scatter wealth'
        
        self.set_frame()

    def scatter_wealth_video_plot(self):
        
        c = (abs(self.actionT[:, 0, self.t-int(self.jump/self.fps):self.t]) > 0).sum(axis=1)
        
        # plotting line
        self.ax.plot(self.x, self.y,'k--', alpha=.2, lw=0.7, label='Initial total wealth') #richness line
        self.ax.scatter( self.Wallet[:, self.t], self.Aequity[:, self.t],
                   c = c, s=2.5, cmap='turbo', alpha=.5)
        
        self.ax.set_xlabel('Wallet ($)')
        self.ax.set_ylabel('Stock value owned ($)')
        self.ax.set_xlim([self.Wallet.min()*0.95-1, self.Wallet.max()*1.05])
        self.ax.set_ylim([self.Aequity.min()*0.95-1, self.Aequity.max()*1.05])
        self.ax.set_xscale('symlog')
        self.ax.set_yscale('symlog')
        #self.ax.loglog()
        
    def QT_radar_plot(self):
    
        categories = ['Bid', 'Hold', 'Ask']
        categories = [*categories, categories[0]]
                
        QT_list = []
        for trader in range(self.I):
            qT_list = [0, 0, 0]
            qT_list[0] = self.QT[trader, self.t, self.sT0, self.sT1, self.sT2, 0]
            qT_list[1] = self.QT[trader, self.t, self.sT0, self.sT1, self.sT2, 1]
            qT_list[2] = self.QT[trader, self.t, self.sT0, self.sT1, self.sT2, 2]
            qT_list = [*qT_list, qT_list[0]]
            QT_list.append( qT_list)

        label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(qT_list))
        
        yMax = self.QT[:, :, self.sT0, self.sT1, self.sT2].max()*1.05
        yMin = self.QT[:, :, self.sT0, self.sT1, self.sT2].min()
        
        for trader in range(self.I):
            
            self.ax.plot(label_loc, QT_list[trader], color=self.colors[trader], alpha=.4)
            self.ax.set_ylim([yMin, yMax] )
        
        #self.ax.title('Restaurant comparison', size=20, y=1.05)
        lines, labels = self.ax.set_thetagrids(np.degrees(label_loc), labels=categories, color='k', alpha=1)
        #self.ax.set_theta_offset(np.deg2rad(90))
        self.ax.set_theta_offset(-np.pi*.30)

    #def QF_space(self, t):

    #    fig = plt.figure(constrained_layout=True, figsize=(10, 7))
    #    gs1 = fig.add_gridspec(nrows=10, ncols=12)
        
    #    self.ax = fig.add_subplot(gs1[:, :11], projection='3d')     # top left     (si
    #    axc = fig.add_subplot(gs1[:, 11:])     # top right     (size: 12x7  - 60x35)

    #    p = self.ax.scatter(self.Q0[:, t],
    #            self.Q1[:, t],
    #            self.Q2[:, t],
    #            c=self.tau,
    #            cmap='jet',
    #            #c=rho)
    #            s = (self.Atotal[:, -1])/50,
    #            #s=(VolumeOwned[-1, :, 0]+10)/10,
    #            alpha=.4)

        #breakpoint()
    #    self.ax.set_xlim([self.Q0.min(axis=(0, 1)), self.Q0.max(axis=(0, 1))])
    #    self.ax.set_ylim([ self.Q1.min(axis=(0, 1)), self.Q1.max(axis=(0, 1))])
    #    self.ax.set_zlim([self.Q2.min(axis=(0, 1))*.9, self.Q2.max(axis=(0, 1))*1.1])
    #    self.ax.set_xlabel('$Q_{a0}$ = 0 \n (Fund.)')
    #    self.ax.set_ylabel('$Q_{a0}$ = 1 \n (Heur.)')
    #    self.ax.set_zlabel('$Q_{a0}$ = 2 \n (Chart.)')

    #    axc.set_axis_off()
        #cb = fig.colorbar(p, ax=axc, label=r'$\tau$' )#r'$g$'+' gesture param.')
    #    cb = fig.colorbar(p, ax=axc, label=r'$\tau$' )#r'$g$'+' gesture param.')
        
    def set_frame(self):
        
        # duration of the video
        duration = 20
        self.jump = np.floor(self.T / duration)
        self.fps = 24
        
        #breakpoint()
        
        # matplot subplot
        
        if self.video in ['learning convergence', 'scatter wealth']:
            self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=300)
        
        elif self.video == 'QT space':
            
            self.fig = plt.figure(constrained_layout=True, figsize=(9, 9))
            gs1 = self.fig.add_gridspec(nrows=10, ncols=12)
            
            self.ax = self.fig.add_subplot(gs1[0:8, :11], projection='3d')     # top left     (size: 12x7  - 60x35)
            #self.axc = self.fig.add_subplot(gs1[0:8, 11:])     # top right     (size: 12x7  - 60x35)
            self.ax2 = self.fig.add_subplot(gs1[8:, :])    # top right    (size: 12x13 - 60x65)
        
        elif self.video == 'QT radar':
        
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = plt.subplot(polar=True)
        
        else:
            self.fig = plt.figure(figsize=(6, 6))
            self.ax1 = plt.subplot(311)
            self.ax2 = plt.subplot(312)
            self.ax3 = plt.subplot(313)
        
        # creating animation
        animation = VideoClip(self.make_frame, duration = duration)
        
        if self.video in ['QT space', 'QT radar']:
            
            animation.write_videofile(self.plotPath+f"/{self.video} - For={self.sT0}, Vol={self.sT1}, PL={self.sT2}.mp4", fps=self.fps)
         
        else:
        
            animation.write_videofile(self.plotPath+f"/{self.video}.mp4", fps=self.fps)

    def make_frame(self, t):

        if self.video in ['learning convergence', 'scatter wealth', 'QT radar']:
            self.ax.clear()
        elif self.video in ['wealth', 'pareto'] :
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
        elif self.video == 'QT space':
            self.ax.clear()    # top left     (size: 12x7  - 60x35)
            #self.axc.clear()    # top right     (size: 12x7  - 60x35)
            self.ax2.clear() 
        
        self.t = int(t*self.jump) #t = 1/fps
        
        #scatter_plot_wealth(t, ax)
        if self.video == 'learning convergence':
            self.convergence_plot()
        elif self.video == 'QT space':
            self.QT_space_video_plot()
        elif self.video == 'wealth':
            self.wealth_video_plot()
        elif self.video == 'pareto':
            self.pareto_plot()
        elif self.video == 'scatter wealth':
            self.scatter_wealth_video_plot()
        elif self.video == 'QT radar':
            self.QT_radar_plot()
        
        if self.video in ['learning convergence', 'scatter wealth']:
            self.ax.set_title(f'Time step: {self.t}')
            plt.tight_layout()
        elif self.video in ['QT space', 'QT radar']:
            self.ax.set_title(f'Time step: {self.t}')
        else:
            self.ax1.set_title(f'Time step: {self.t}')
        
        #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        #ax.color 
        
        # returning numpy image
        return mplfig_to_npimage(self.fig)

