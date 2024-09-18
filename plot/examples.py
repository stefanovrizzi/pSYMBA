#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:15:18 2023

@author: svrizzi
"""


#%%

import sys
sys.path.append('..') if '..' not in sys.path else None  #go to parent folder

import numpy as np
from initialise.time_parameters import tInit

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching
import os

import pandas as pd
import scipy.stats as stats 
from scipy.stats import skew
import statsmodels.api as sm
from scipy import signal

import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns

#%%

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

#%%

class PlotExamples:

    def __init__(self, datasetFolder):
                
        self.datasetFolder = datasetFolder
       
        self.stylisedFactsNames = [r'$\mu$', r'$\sigma^{2}$', 'skw', 'kurt', 'AC1', 'AC|r|100']
        self.numberStylisedFacts = len(self.stylisedFactsNames)
        
        self.load_general_info()

        self.plotPathDatasets = self.fetch.plotFolderPath+'/'+datasetFolder
        os.makedirs(self.plotPathDatasets) if not os.path.isdir(self.plotPathDatasets) else None #create Data folder for all data files, in case it does not exist
        
        self.load_hyperparameters()

        self.cmapName='turbo'
        
        self.plot_equilibrium_price() if self.hyperparameterName in ['Liquidity balance', 'InitialVolumeOwned'] else None
            
        self.load_examples()         
        self.plot()
    
    def stylised_facts(self, marketVariable):
        
        sf = np.zeros((len(marketVariable.columns), self.numberStylisedFacts))
        #breakpoint()
        sf[:, 0] = marketVariable.mean()
        sf[:, 1] = marketVariable.var()
        sf[:, 2] = skew(marketVariable)
        sf[:, 3] = marketVariable.kurt()        
        sf[:, 4] = marketVariable.apply(lambda x : sm.tsa.acf(x, nlags=1)[1])
        sf[:, 5] = marketVariable.apply(lambda x : (sm.tsa.acf(x.abs(), nlags=100)[1:]).sum())
        
        return sf
    
    def load_examples(self):

        #Macro-variables
        self.Price = np.zeros((self.examplesReady*self.S, self.T-tInit))
        self.Volume = np.zeros((self.examplesReady*self.S, self.T-tInit))
        self.StylisedFactsReturns = np.zeros((self.examplesReady*self.S, self.numberStylisedFacts)) #stylised facts returns
        self.StylisedFactsVolumes = np.zeros((self.examplesReady*self.S, self.numberStylisedFacts)) #stylised facts volumes
        
        #agent parameters
        self.g = np.zeros((self.examplesReady*self.S, self.I))
        self.tau = np.zeros((self.examplesReady*self.S, self.I))

        self.AvailableMargin = np.zeros((self.examplesReady, self.S, self.I))

        for exampleNumber in range(self.examplesReady):

            store = FileStorage(datasetFolder=self.datasetFolder)
            store.set_example(exampleNumber)
            fetch = FileFetching(store)
            
            data = LoadData(fetch)
            
            self.StylisedFactsReturns[self.S*exampleNumber:self.S*exampleNumber+self.S] = self.stylised_facts(data.rdf)
            self.StylisedFactsVolumes[self.S*exampleNumber:self.S*exampleNumber+self.S] = self.stylised_facts(data.vdf)

            self.Price[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.Price
            self.Volume[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.Volume
           
            self.AvailableMargin[exampleNumber] = data.AvailableMargin[:, :]
           
            self.g[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.g
            self.tau[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.tau
           
        self.g = self.g*100

        #breakpoint()
        
        self.dfStylisedFactsReturns = pd.DataFrame(data=self.StylisedFactsReturns, columns=self.stylisedFactsNames)
        
        self.dfStylisedFactsVolumes = pd.DataFrame(data=self.StylisedFactsVolumes, columns=self.stylisedFactsNames)
        #self.dfStylisedFactsVolumes.fillna(0, inplace=True)
        
    def load_general_info(self):
    
        self.load_input_parameters()
        self.load_LSE_data()
        self.StylisedFactsLSEreturns = self.stylised_facts(self.LSEreturns)
        self.StylisedFactsLSEvolumes = self.stylised_facts(self.LSEvolumes)
        
        self.dfStylisedFactsLSEReturns = pd.DataFrame(data=self.StylisedFactsLSEreturns, columns=self.stylisedFactsNames, index=self.LSEreturns.columns)
    
    def load_input_parameters(self):
        
        store = FileStorage(datasetFolder=self.datasetFolder)
        self.examplesReady = store.check_flag()+1   
        store.set_example(0)
        
        #breakpoint()
        
        self.fetch = FileFetching(store)
        
        #Input variables
        self.InputParameters = self.fetch.input_parameters()
        self.Ttraining = self.InputParameters['Ttraining']
        self.T = self.InputParameters['T']
        self.I = self.InputParameters['I']
        self.J = self.InputParameters['J']
        self.S = self.InputParameters['S']
        self.LSEstock = self.InputParameters['LSEstock']
    
    def load_LSE_data(self):
        
        ### Empirical LSE data ###
        try:
            self.LSEprices = pd.read_csv('LSE_data/LSE_data_closing_prices.csv', index_col=False)
        except:
            self.LSEprices = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)

        self.LSEreturns = np.log10(self.LSEprices).diff(periods=1, axis=0).iloc[1:]
        
        try:
            self.LSEvolumes = pd.read_csv('LSE_data/LSE_data_volumes.csv', index_col=False)
        except:
            self.LSEvolumes = pd.read_csv('plot/LSE_data/LSE_data_volumes.csv', index_col=False) 

    def load_hyperparameters(self):
    
        self.hyperparameters = self.fetch.parameters(parameterName='hyperparameters')

        self.hyperparameters.pop('distribution', None)
        
        self.hyperparameterName = list(self.hyperparameters.keys())[0] #get only first key (first hyperparameter, the one that you tweaked)
        
        self.parameterName = self.parameter_name(self.hyperparameterName)
        
        self.hyperparametersData = np.repeat(self.hyperparameters[self.hyperparameterName], self.S, axis=0)
        
        self.hyperparameterName = self.parameterName if 'alpha' in self.hyperparameterName[:5] else self.hyperparameterName
        
        self.dfhyperparameters = pd.DataFrame(data=self.hyperparametersData, columns=[self.hyperparameterName])
        
        self.hyperparameterRange = self.dfhyperparameters[self.hyperparameterName].unique()
        
        if len(self.hyperparameters.keys())>1: #more than one hyperparameter in list
            self.secondaryHyperparameterName = list(self.hyperparameters.keys())[1]
            self.secondaryHyperparameter = set(list(self.hyperparameters.values())[1])
            self.secondaryParameterName = self.parameter_name(self.secondaryHyperparameterName)
            self.secondaryHyperparameterName = self.secondaryParameterName if 'alpha' in self.secondaryHyperparameterName[:5] else self.secondaryHyperparameterName
        else:
            self.secondaryHyperparameterName = ''
            self.secondaryParameterName = ''            
        
    def parameter_name(self, hyperparameterName):
               
        if 'g' in hyperparameterName[:1]:
            
            parameterName = 'g'
            
        elif 'tau' in hyperparameterName[:3]:
        
            parameterName = 'tau'
            
        elif 'alpha' in hyperparameterName[:5]:
            
            split = hyperparameterName.split('alpha')            
            subscript = '+' if split[1] == 'Conf' else '-' if split[1] == 'Disc' else None
            parameterName = r'$\alpha_{}$'.format(subscript)
            #hyperparameterName = parameterName
            
        elif 'Alpha' in hyperparameterName[1:6]:
            
            parameterName = hyperparameterName[1:]
        
        elif 'rho' in hyperparameterName[1:]:
            
            parameterName = r'$\rho$'
        
        elif hyperparameterName in ['InitialVolumeOwned', 'bFlat', 'Liquidity balance', 'distribution', 'kernel']:
            
            parameterName = hyperparameterName
            
        #breakpoint()
            
        return parameterName
        
        
    ##############################
    ############ PLOTS ###########
    ##############################
    
    def plot_equilibrium_price(self):
    
        #Macro-variables
        self.Price = np.zeros((self.examplesReady, self.Ttraining-tInit))
        #self.Volume = np.zeros((self.examplesReady, self.T-tInit))

        for exampleNumber in range(self.examplesReady):

            store = FileStorage(datasetFolder=self.datasetFolder)
            store.set_example(exampleNumber)
            fetch = FileFetching(store)
            
            self.Price[exampleNumber] = fetch.saved('Price', verbose=False)[0, tInit:]
            
        meanTimeWindow = 252
            
        plt.figure(figsize=(6,6))
        plt.scatter(self.hyperparameters[self.hyperparameterName], self.Price[:, -meanTimeWindow:].mean(axis=1), c=self.hyperparameters[self.hyperparameterName], cmap=self.cmapName)
        plt.xlabel(f'{self.hyperparameterName}')
        plt.ylabel(f'Mean stock price ($) over {meanTimeWindow} time steps')
        plt.xscale('symlog') if 'InitialVolumeOwned' in self.parameterName else None
        plt.savefig(self.plotPathDatasets+f'/{self.hyperparameterName}_vs_equilibrium_price.png', dpi=200)
        plt.close()
            

    def plot(self):
        
        self.set_colormap()
        self.hyperparameters_vs_stylised_facts('returns')
        self.hyperparameters_vs_stylised_facts('volumes')
        self.example_trace('price')
        self.example_trace('volume')
        #self.agent_parameter_sampling()

    def set_colormap(self):
        
        self.cmap = plt.get_cmap(self.cmapName)
        self.norm = plt.Normalize(0, self.examplesReady)
        
    def hyperparameters_vs_stylised_facts(self, macrovariableName):

        # HP vs SF returns        
        df = pd.concat([self.dfStylisedFactsReturns, self.dfhyperparameters], axis=1) if macrovariableName == 'returns' else pd.concat([self.dfStylisedFactsVolumes, self.dfhyperparameters], axis=1)
        #StylisedFactsLSE = self.StylisedFactsLSEreturns if macrovariableName == 'returns' else self.StylisedFactsLSEvolumes

        fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharex=True)
        axs = axs.ravel()

        gs = GridSpec(3, 3, figure=fig)
        [ax.remove() for ax in axs[:3]] 
        ax = fig.add_subplot(gs[0, :])

        #N = min(4, self.examplesReady-1)
        #for n in np.arange(0, self.examplesReady+1, N): #N examples of distributions
        
        #    hyperparameter = self.hyperparameterRange[n]
        #    color = self.cmap(self.norm(n))
            #ax = axs[0]
            #ax = plt.subplots(3,3,[1:4])
            
        #    if self.parameterName == 'g':           
        #        x = np.linspace(0, 35, 200)
        #        y = stats.gamma.pdf(x, a=hyperparameter, scale=1) #check scale and 1/beta https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

        #    if self.parameterName == 'tau':           
        #        x = np.linspace(0, 35, 200)
        #        y = stats.gamma.pdf(x, a=2.5, scale=hyperparameter)
            
        #    if 'rho' in self.parameterName:           
        #        x = np.linspace(0, 1, 100)
        #        y = stats.beta.pdf(x, a=hyperparameter, b=1) #check scale and 1/beta https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
                
        #    elif 'Alpha' in self.parameterName:
        #        x = np.linspace(0, 1, 200)
        #        y = stats.beta.pdf(x, a=hyperparameter, b=1)
                
        #    elif 'alpha' in self.parameterName:
        #        m = self.examplesReady+1 #granularity
        #        x = np.linspace(0, 1, self.examplesReady*m+(m-1))
        #        y = signal.unit_impulse(self.examplesReady*m+(m-1), n*m)
                
        #    elif self.parameterName in ['Liquidity balance', 'bFlat']:
        #        m = self.examplesReady+1 #granularity
        #        x = np.linspace(self.hyperparameterRange.min(), self.hyperparameterRange.max(), self.examplesReady*m+(m-1))
        #        y = signal.unit_impulse(self.examplesReady*m+(m-1), n*m)
                
        #    elif 'InitialVolumeOwned' in self.parameterName:
        #        m = self.examplesReady+1 #granularity
        #        x = np.logspace(np.log10(self.hyperparameterRange.min()), np.log10(self.hyperparameterRange.max()), self.examplesReady*m+(m-1))
        #        y = signal.unit_impulse(self.examplesReady*m+(m-1), n*m)
                
        #    ax.plot(x, y, color=color, alpha=.6, linewidth=2.3, label=f"{self.hyperparameterName} = {round(hyperparameter, 2)}")
        #    ax.set_xscale('symlog') if 'InitialVolumeOwned' in self.parameterName else None
            
        #ax.set_xlabel(self.parameterName)
        #ax.set_ylabel('Density')
        #plt.legend()

        for stylisedFactNumber, stylisedFact in enumerate(self.stylisedFactsNames):
            ax = axs[3+stylisedFactNumber]
            ax.axhline(y=0, color='k', alpha=.2)
            #breakpoint()
            NaNwarning = f'Replaced NaN for {stylisedFact}'
            print(NaNwarning) if df.isnull().values.any() else None
            df.fillna(0, inplace=True)
            graph = sns.scatterplot(data=df, x=self.hyperparameterName, y=stylisedFact, ax=ax, alpha=.4, c=df[self.hyperparameterName], cmap=self.cmap)
            sns.scatterplot(data=df.groupby(self.hyperparameterName).mean(), x=self.hyperparameterName, y=stylisedFact, ax=ax, color='k', marker='.')
            
            if macrovariableName == 'returns':
                
                graph.axhline(self.dfStylisedFactsLSEReturns.loc[self.LSEstock][stylisedFact], linestyle=':', linewidth=.7, alpha=1, c=[.6, .1, .1], label=self.LSEstock)
                graph.axhline(self.dfStylisedFactsLSEReturns[stylisedFact].median(), linestyle=':', linewidth=.7, alpha=1, c=[.1, .1, .6], label='LSE median')
                graph.legend() if stylisedFactNumber == len(self.stylisedFactsNames) - 1 else None
                
            ax.yaxis.major.formatter.set_powerlimits((0,0))
            
            ax.set_xscale('symlog') if 'InitialVolumeOwned' in self.parameterName else None
            
            ax.set_title(NaNwarning) if df.isnull().values.any() else None
            
        title = f'Hyperparameter of interest: {self.hyperparameterName}'
        #title = title+'- secondary hyperparameter: {self.secondaryHyperparameterName} = {self.secondaryHyperparameter}' if
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.plotPathDatasets+f'/parameter_vs_stylised_facts_{macrovariableName}.png', dpi=200)
        plt.close()
        
    def example_trace(self, macrovariableName):
        
        macrovariable = self.Price if macrovariableName == 'price' else self.Volume if macrovariableName == 'volume' else None
        n = np.ceil(np.sqrt(self.examplesReady)).astype(int)
        n = max(n, 2)
        fig, axs = plt.subplots(n, n, figsize=(12, 9), sharex=True)
        axs = axs.ravel()

        for exampleNumber in range(self.examplesReady):
            
            ax = axs[exampleNumber]
            color = self.cmap(self.norm(exampleNumber))
            
            #[ax.plot(Price[S*exampleNumber+s], alpha=.1, color=color) for s in range(S)]
            [ax.plot( macrovariable[self.S*exampleNumber+s], alpha=1, color=color) for s in [0]]
            ax.set_ylim([macrovariable.min(), macrovariable.max()])
            
            ax.set_yscale('symlog') if 'InitialVolumeOwned' in self.parameterName and macrovariableName == 'volume' else None
            
        fig.supxlabel('Time steps (testing phase)')
        ylabel = f'Stock {macrovariableName}'
        ylabel += ' ($)' if macrovariableName == 'price' else ''
        fig.supylabel(ylabel)
        title = f'Hyperparameter of interest: {self.hyperparameterName}'
        #title = title+f'- secondary hyperparameter: {self.secondaryHyperparameterName} = {self.secondaryHyperparameter}'
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.plotPathDatasets+f'/example_of_{macrovariableName}_trace.png', dpi=200)
        plt.close()
        
# =============================================================================
#     def agent_parameter_sampling(self):
#         
#         # Did we sample agent parameters well? Check
# 
#         fig, axs = plt.subplots(6, 5, figsize=(12, 12), sharex=True)
#         axs = axs.ravel()
# 
#         x = np.linspace(0, 45, 200)
# 
#         for exampleNumber in range(self.examplesReady):
#             ax = axs[exampleNumber]
#             
#             hyperparameterValue = df[hyperparameter].unique()[exampleNumber]
#             color = self.cmap(self.norm(hyperparameterValue))
#             
#             y = stats.gamma.pdf(x, a=hyperparameterValue, scale=1)
#             ax.plot(x, y, color=color, alpha=1, linewidth=2.3, label=f"{hyperparameter} = {hyperparameterValue}")
#             
#             #[ax.plot(Price[S*exampleNumber+s], alpha=.1, color=color) for s in range(S)]
#             #[ax.hist(hp_agents[S*exampleNumber+s], density=True, alpha=0.01, color=color) for s in range(S)]
#             [ax.hist(hp_agents[self.S*exampleNumber+s], density=True, alpha=0.4, color=color) for s in [0]]
#             ax.set_xlim([parameter.min(), parameter.max()])
#             
#         #fig.supxlabel(r'$\tau$')
#         fig.supxlabel(parameterName)
#         fig.supylabel('Frequency')
#         plt.tight_layout()
#         plt.savefig(f'{parameterName}_sampling.png', dpi=200)
#         plt.close()
# =============================================================================

class LoadData:
    
    def __init__(self, fetch):   
    
        self.InputParameters = fetch.input_parameters()
        self.T = self.InputParameters['T']
        self.I = self.InputParameters['I']
        self.J = self.InputParameters['J']
        self.S = self.InputParameters['S']
        
        #RL parameters
        self.RLparameters = fetch.parameters('RL')
        self.CognitiveTraits = self.InputParameters['CognitiveTraits']
        
        self.beta = self.RLparameters['beta']
        self.alphaConf = self.RLparameters['alphaConf']
        
        if 'optimistic bias' in self.CognitiveTraits:
            self.alphaDisc = self.RLparameters['alphaDisc']
        
        #SYMBA parameters
        self.SYMBAparameters = fetch.parameters('SYMBA')
        #self.FundamentalValues = self.SYMBAparameters['FundamentalValues'][:, 0] #stock 0
        #self.rho = self.SYMBAparameters['rho']
        self.tau = self.SYMBAparameters['tau']
        self.g = self.SYMBAparameters['g']
        
        #Data generated from simulation
        #self.data = fetch.saved_data(s)
        
        #Market variables
        self.Price = np.zeros((self.S, self.T-tInit))
        self.Volume = np.zeros((self.S, self.T-tInit))
        self.AvailableMargin = np.zeros((self.S, self.I))
    
        for s in range(self.S):
            self.Price[s] = fetch.saved('Price', s=s, verbose=False)[0, tInit:]
            self.Volume[s] = fetch.saved('Volume', s=s, verbose=False)[0, tInit:]
            self.AvailableMargin[s] = fetch.saved('AvailableMargin', s=s, verbose=False)[:, tInit-1]

        self.returns = np.log10(self.Price[:, 1:]) - np.log10(self.Price[:, :-1]) #returns
        self.rdf = pd.DataFrame(data=self.returns.T, columns=list(range(self.S)))
        
        self.vdf = pd.DataFrame(data=self.Volume.T, columns=list(range(self.S)))
