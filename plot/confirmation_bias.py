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

import math

#%%

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

#%%

class PlotConfirmationBias:

    def __init__(self, datasetFolder):
                
        self.datasetFolder = datasetFolder
       
        self.stylisedFactsNames = [r'$\mu$', r'$\sigma^{2}$', 'skw', 'kurt', 'AC1', 'AC|r|100']
        self.numberStylisedFacts = len(self.stylisedFactsNames)
        
        self.load_general_info()

        self.plotPathDatasets = self.fetch.plotFolderPath+'/'+datasetFolder
        os.makedirs(self.plotPathDatasets) if not os.path.isdir(self.plotPathDatasets) else None #create Data folder for all data files, in case it does not exist
        
        self.load_hyperparameters()
        
        #self.plot_equilibrium_price() if self.hyperparameterName in ['Liquidity balance', 'InitialVolumeOwned'] else None
            
        self.load_examples()       
        self.plot()
    
    def stylised_facts(self, marketVariable):
        
        sf = np.zeros((len(marketVariable.columns), self.numberStylisedFacts))

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
        #self.dfReturns = pd.DataFrame()
        
        self.StylisedFactsReturns = np.zeros((self.examplesReady*self.S, self.numberStylisedFacts)) #stylised facts returns
        self.StylisedFactsVolumes = np.zeros((self.examplesReady*self.S, self.numberStylisedFacts)) #stylised facts volumes
        
        #agent parameters
        self.g = np.zeros((self.examplesReady*self.S, self.I))
        self.tau = np.zeros((self.examplesReady*self.S, self.I))
        self.alphaConf = np.zeros((self.examplesReady*self.S, self.I))
        self.alphaDisc = np.zeros((self.examplesReady*self.S, self.I))

        for exampleNumber in range(self.examplesReady):

            store = FileStorage(datasetFolder=self.datasetFolder)
            store.set_example(exampleNumber)
            fetch = FileFetching(store)
            
            data = LoadData(fetch)
            
            self.StylisedFactsReturns[self.S*exampleNumber:self.S*exampleNumber+self.S] = self.stylised_facts(data.rdf)
            self.StylisedFactsVolumes[self.S*exampleNumber:self.S*exampleNumber+self.S] = self.stylised_facts(data.vdf)
            
            #breakpoint()
            #self.dfReturns = pd.concat([self.dfReturns, data.rdf])
            
            self.Price[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.Price
            self.Volume[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.Volume
                      
            self.g[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.g
            self.tau[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.tau
            
            self.alphaConf[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.alphaConf
            
            self.alphaDisc[self.S*exampleNumber:self.S*exampleNumber+self.S] = data.alphaDisc if 'optimistic bias' in data.CognitiveTraits else data.alphaConf
                       
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
        
        self.store = FileStorage(datasetFolder=self.datasetFolder)
        self.examplesReady = self.store.check_flag()+1   
        self.store.set_example(0)
        
        #breakpoint()
        
        self.fetch = FileFetching(self.store)
        
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

        self.parameterName = list(self.hyperparameters.keys())[0]

        #self.hyperparameters.pop('distribution', None)
        #self.hyperparameters.pop('kernel', None)
        
        if self.parameterName == 'aAlphaConf':
            
            self.dfhyperparameters = pd.read_csv(self.store.DataFolderPath+'/hyperparameters.csv', index_col=False)
            #self.dfhyperparameters = pd.DataFrame.from_dict(self.hyperparameters)
            
            #breakpoint()
            self.dfhyperparameters['alphaConf'] = self.dfhyperparameters['aAlphaConf'] / (self.dfhyperparameters['aAlphaConf']+self.dfhyperparameters['bAlphaConf'])
            self.dfhyperparameters['alphaConf'] = self.dfhyperparameters['alphaConf']*.1
            
            self.dfhyperparameters['alphaDisc'] = self.dfhyperparameters['aAlphaConf.1'] / (self.dfhyperparameters['aAlphaConf.1']+self.dfhyperparameters['bAlphaConf.1'])
            self.dfhyperparameters['alphaDisc'] = self.dfhyperparameters['alphaDisc']*.1
            
            #self.dfhyperparameters['alphaDisc'] = self.dfhyperparameters['aAlphaDisc'] / (self.dfhyperparameters['aAlphaDisc']+self.dfhyperparameters['bAlphaDisc'])
            
        else:
             
            self.dfhyperparameters = pd.DataFrame.from_dict(self.hyperparameters)
        #self.dfhyperparameters = pd.read_csv('hyperparameters.csv')

        self.dfhyperparameters['Confirmation bias'] =  self.dfhyperparameters['alphaConf'] - self.dfhyperparameters['alphaDisc']        
        #self.dfhyperparameters['Confirmation bias'].fillna(0, inplace=True)

        self.hyperparameterRange = self.dfhyperparameters['Confirmation bias'].unique() 
        
        self.dfhyperparameters.drop(['distribution', 'kernel'], axis=1, inplace=True)
        
        self.dfhyperparameters = self.dfhyperparameters.loc[self.dfhyperparameters.index.repeat(self.S)].reset_index(drop=True)
        
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
    
    def simulation_analysis(self):
    
        for exampleNumber in range(self.examplesReady):

            store = FileStorage(datasetFolder=self.datasetFolder)
            store.set_example(exampleNumber)
            fetch = FileFetching(store)
            data = LoadData(fetch)
            
            examplePlotFolder = self.plotPathDatasets+f'/ex{exampleNumber}'
    
            data.market_dynamics(examplePlotFolder)
            data.market_dynamics(examplePlotFolder, 'testing')
            data.attempts_vs_wealth(examplePlotFolder)
            data.QF_dynamics(examplePlotFolder)
    
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
        
        #self.hyperparameters_vs_stylised_facts_REPORT('returns')
        #self.hyperparameters_vs_stylised_facts_REPORT('volumes')

        self.hyperparameters_vs_stylised_facts_bubbleview_REPORT('returns')
        #self.hyperparameters_vs_stylised_facts_bubbleview_REPORT('volumes')
        
        #self.hyperparameters_vs_stylised_facts('returns')
        #self.hyperparameters_vs_stylised_facts('volumes')
        ##self.hyperparameters_vs_stylised_facts_bubbleview('returns', 'low')
        ##self.hyperparameters_vs_stylised_facts_bubbleview('volumes', 'low')
        #self.stylised_facts_distributions('returns')
        #self.stylised_facts_distributions('volume')
        #self.hyperparameters_vs_stylised_facts_bubbleview('returns', 'high')
        #self.hyperparameters_vs_stylised_facts_bubbleview('volumes', 'high')
        #self.example_trace('price')
        #self.example_trace('volume')
        #self.agent_parameter_sampling()
            
        #self.simulation_analysis()

    def set_colormap(self):
        
        cmapName='turbo'
        self.cmap = plt.get_cmap(cmapName)
        self.norm = plt.Normalize(0, self.examplesReady)
        
    #def 
        
    #    g = sns.pairplot(iris, hue="Confirmation bias", palette="coolwarm", diag_kind="kde", height=2.5)

############################################################################

    def hyperparameters_vs_stylised_facts_REPORT(self, macrovariableName):
        
        # HP vs SF returns
        
        df = pd.concat([self.dfStylisedFactsReturns, self.dfhyperparameters], axis=1) if macrovariableName == 'returns' else pd.concat([self.dfStylisedFactsVolumes, self.dfhyperparameters], axis=1)

        df.rename(columns={'alphaConf':'$\alpha_{+}$', 'alphaDisc':'$\alpha_{-}$'}, inplace=True)
        
        fig, axs = plt.subplots(3, 2, figsize=(8, 9), sharex=True)
        axs = axs.ravel()

        for stylisedFactNumber, stylisedFact in enumerate(self.stylisedFactsNames):
            ax = axs[stylisedFactNumber]
            ax.axhline(y=0, color=[.6, .6, .6], lw=.5, alpha=.2)

            NaNwarning = f'Replaced NaN for {stylisedFact}'
            print(NaNwarning) if df[stylisedFact].isnull().values.any() else None
            df[stylisedFact].fillna(0, inplace=True)
            
            graph = sns.scatterplot(data=df, x='Confirmation bias', y=stylisedFact, ax=ax, alpha=.4, color=[.5, 0, 0], s=df['$\alpha_{+}$']*4+5, label='pSYMBA')
            graph = sns.scatterplot(data=df.groupby(['$\alpha_{+}$', '$\alpha_{-}$']).mean(), x='Confirmation bias', y=stylisedFact, ax=ax, color='k', marker='o', s=20, label='pSYMBA mean')
            graph.legend_.remove()                       
            
            if macrovariableName == 'returns':
                
                graph.axhline(self.dfStylisedFactsLSEReturns.loc[self.LSEstock][stylisedFact], linestyle='-', linewidth=1, alpha=1, c=[0, 0, .5], label=self.LSEstock)
                graph.axhline(self.dfStylisedFactsLSEReturns[stylisedFact].mean(), linestyle='-', linewidth=1, alpha=1, c=[.5, .5, .5], label='LSE mean')
                #graph.legend() if stylisedFactNumber == len(self.stylisedFactsNames) - 1 else None
            
                ax.set_ylim([-2.5, None]) if stylisedFact == 'skw' else ax.set_ylim([-0.01, self.dfStylisedFactsLSEReturns[stylisedFact].mean()*1.05]) if stylisedFact == 'kurt' else None
            
            res = stats.linregress(df['Confirmation bias'], df[stylisedFact])
            
            print(stylisedFact, ' p: ', res.pvalue.round(2), ', r: ', res.rvalue.round(3))
            
            x = [-1, 1] #ax.get_xlim()
            x = np.linspace(x[0], x[-1], 3)
                       
            if res.pvalue < 0.05 / len(self.stylisedFactsNames):
                ax.plot(x, res.intercept + res.slope*x, color=[.5, 0, 0], linestyle='-', alpha=1, lw=2, label='linear regression (*)')
            else:
                ax.plot(x, res.intercept + res.slope*x, color=[.5, 0, 0], linestyle='--', alpha=1, lw=1, label='linear regression (NS)')
            
            ax.set_xlim([-1.05, 1.05])
            #ax.yaxis.major.formatter()
            #ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
            ax.yaxis.major.formatter.set_powerlimits((-4,1))
            #ax.set_xscale('symlog', base=2) if np.log10(df['Confirmation bias'].max())-np.log10( df['Confirmation bias'][df['Confirmation bias']>0].min() ) >= 2 else ax.yaxis.major.formatter.set_powerlimits((0,0))
            
            #ax.set_yscale('symlog', base=10)
            
            ax.set_title(NaNwarning) if df.isnull().values.any() else None
            
            if stylisedFactNumber == len(self.stylisedFactsNames) - 1:
            
                from matplotlib.lines import Line2D
                handles, labels = plt.gca().get_legend_handles_labels() 
                
                if res.pvalue >= 0.05 / len(self.stylisedFactsNames):
                    line = Line2D([0], [0], label='linear regression (*)', color=[.5, 0, 0], linestyle='-', alpha=1, lw=2)
                else:
                    line = Line2D([0], [0], label='linear regression (NS)', color=[.5, 0, 0], linestyle='-', alpha=1, lw=1)
                handles.extend([line])
                #fig.legend(handles=handles, bbox_to_anchor=(0.98, .6))
                
                if macrovariableName == 'returns':
                    fig.legend(handles=handles, bbox_to_anchor=(0.8, .98), ncol=3)
                else:
                    fig.legend(handles=handles, bbox_to_anchor=(0.7, .98), ncol=2)      
        
        plt.tight_layout()
        #fig.subplots_adjust(left=0.1, right=0.72, wspace=0.34, hspace=0.15)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.85, wspace=0.34, hspace=0.15)
        
        title = '' #f'Hyperparameter of interest: {self.hyperparameterName}'
        #title = title+'- secondary hyperparameter: {self.secondaryHyperparameterName} = {self.secondaryHyperparameter}' if
        plt.suptitle(title)
        
        plt.savefig(self.plotPathDatasets+f'/parameter_vs_stylised_facts_{macrovariableName}_REPORT.pdf', dpi=200)
        plt.close()

    def hyperparameters_vs_stylised_facts_bubbleview_REPORT(self, macrovariableName):
        
        ### IT WOULD BE GOOD TO ADD A STATISTICAL TEST AGAINST LSE or Wieners
        
        # HP vs SF returns        
        dfTemp = pd.concat([self.dfStylisedFactsReturns, self.dfhyperparameters], axis=1) if macrovariableName == 'returns' else pd.concat([self.dfStylisedFactsVolumes, self.dfhyperparameters], axis=1)
        
        #df.rename(columns={'alphaConf':r'$\alpha_{+}$', 'alphaDisc':r'$\alpha_{-}$'}, inplace=True)
        
        df = dfTemp.groupby(['alphaConf', 'alphaDisc']).mean()
        
        from plot.Dunnett import dunnett
        from scipy.stats import yeojohnson
        
        if macrovariableName == 'returns':
                     
            m = len(df.index)*len(self.stylisedFactsNames)
            
            #Sidak correction (should already be included in Dunnett)
            alpha = 0.05
            #alpha = 1-(1-alpha)**(1/m)
                   
            for stylisedFact in self.stylisedFactsNames:

                #df[stylisedFact+'_pvalue'] = dfTemp.groupby(['alphaConf', 'alphaDisc'])[stylisedFact].apply(lambda x: stats.anderson_ksamp([ x, self.dfStylisedFactsLSEReturns[stylisedFact]]).pvalue < alpha )
        
                dfTemp[stylisedFact+'_transformed'] = dfTemp[stylisedFact]

### Trasform data ###
                #for (alphaConf, alphaDisc) in dfTemp.set_index(['alphaConf', 'alphaDisc']).index.unique():
                
                #    dfTemp.loc[(dfTemp.alphaConf == alphaConf)*(dfTemp.alphaDisc == alphaDisc), stylisedFact+'_transformed'] = yeojohnson( dfTemp[(dfTemp.alphaConf == alphaConf)*(dfTemp.alphaDisc == alphaDisc)][stylisedFact+'_transformed'] )[0]
                
                stylisedFactsTemp = dfTemp.groupby(['alphaConf', 'alphaDisc'])[stylisedFact].apply(pd.Series.tolist).tolist()
                
                #Anova
            
                from scipy.stats import f_oneway
            
            
                _, p = f_oneway(stylisedFactsTemp[0],
                stylisedFactsTemp[1],
                stylisedFactsTemp[2],
                stylisedFactsTemp[3],
                stylisedFactsTemp[4],
                stylisedFactsTemp[5],
                stylisedFactsTemp[6],
                stylisedFactsTemp[7],
                stylisedFactsTemp[8],
                stylisedFactsTemp[9],
                stylisedFactsTemp[10],
                stylisedFactsTemp[11],                
                stylisedFactsTemp[12],
                stylisedFactsTemp[13],
                stylisedFactsTemp[14],
                stylisedFactsTemp[15],
                stylisedFactsTemp[16],
                stylisedFactsTemp[17],
                stylisedFactsTemp[18],
                stylisedFactsTemp[19],
                stylisedFactsTemp[20],
                stylisedFactsTemp[21],
                stylisedFactsTemp[22],
                stylisedFactsTemp[23],
                stylisedFactsTemp[24],
                stylisedFactsTemp[25],
                stylisedFactsTemp[26],
                stylisedFactsTemp[27],
                stylisedFactsTemp[28],
                stylisedFactsTemp[29],
                stylisedFactsTemp[30],
                stylisedFactsTemp[31],                
                stylisedFactsTemp[32],
                stylisedFactsTemp[33],
                stylisedFactsTemp[34],
                stylisedFactsTemp[35],
                stylisedFactsTemp[36],
                stylisedFactsTemp[37],
                stylisedFactsTemp[38],
                stylisedFactsTemp[39],
                stylisedFactsTemp[40],
                stylisedFactsTemp[41],                
                stylisedFactsTemp[42],
                stylisedFactsTemp[43],
                stylisedFactsTemp[44],
                stylisedFactsTemp[45],
                stylisedFactsTemp[46],
                stylisedFactsTemp[47],
                stylisedFactsTemp[48],
                stylisedFactsTemp[49],
                stylisedFactsTemp[50],
                stylisedFactsTemp[51],                
                stylisedFactsTemp[52],
                stylisedFactsTemp[53],
                stylisedFactsTemp[54],
                stylisedFactsTemp[55],
                stylisedFactsTemp[56],
                stylisedFactsTemp[57],
                self.dfStylisedFactsLSEReturns[stylisedFact])
                
                
                if p>=0.05: #no difference between groups, so pSYMBA supposedly captures LSE
                    df[stylisedFact+'_pvalue'] = (p<0.05)
                    print(stylisedFact, p)
                else: #pSYMBA may be different to LSE or among its simulations: post-hoc Dunnett's test
                
                
                    pvalues = dunnett(
                    stylisedFactsTemp[0],
                    stylisedFactsTemp[1],
                    stylisedFactsTemp[2],
                    stylisedFactsTemp[3],
                    stylisedFactsTemp[4],
                    stylisedFactsTemp[5],
                    stylisedFactsTemp[6],
                    stylisedFactsTemp[7],
                    stylisedFactsTemp[8],
                    stylisedFactsTemp[9],
                    stylisedFactsTemp[10],
                    stylisedFactsTemp[11],                
                    stylisedFactsTemp[12],
                    stylisedFactsTemp[13],
                    stylisedFactsTemp[14],
                    stylisedFactsTemp[15],
                    stylisedFactsTemp[16],
                    stylisedFactsTemp[17],
                    stylisedFactsTemp[18],
                    stylisedFactsTemp[19],
                    stylisedFactsTemp[20],
                    stylisedFactsTemp[21],
                    stylisedFactsTemp[22],
                    stylisedFactsTemp[23],
                    stylisedFactsTemp[24],
                    stylisedFactsTemp[25],
                    stylisedFactsTemp[26],
                    stylisedFactsTemp[27],
                    stylisedFactsTemp[28],
                    stylisedFactsTemp[29],
                    stylisedFactsTemp[30],
                    stylisedFactsTemp[31],                
                    stylisedFactsTemp[32],
                    stylisedFactsTemp[33],
                    stylisedFactsTemp[34],
                    stylisedFactsTemp[35],
                    stylisedFactsTemp[36],
                    stylisedFactsTemp[37],
                    stylisedFactsTemp[38],
                    stylisedFactsTemp[39],
                    stylisedFactsTemp[40],
                    stylisedFactsTemp[41],                
                    stylisedFactsTemp[42],
                    stylisedFactsTemp[43],
                    stylisedFactsTemp[44],
                    stylisedFactsTemp[45],
                    stylisedFactsTemp[46],
                    stylisedFactsTemp[47],
                    stylisedFactsTemp[48],
                    stylisedFactsTemp[49],
                    stylisedFactsTemp[50],
                    stylisedFactsTemp[51],                
                    stylisedFactsTemp[52],
                    stylisedFactsTemp[53],
                    stylisedFactsTemp[54],
                    stylisedFactsTemp[55],
                    stylisedFactsTemp[56],
                    stylisedFactsTemp[57],
                    control=self.dfStylisedFactsLSEReturns[stylisedFact]).pvalue
                    print('ANOVA', p, 'Dunnett', stylisedFact, min(pvalues), max(pvalues))
                    pvalues = pvalues < alpha
                    df[stylisedFact+'_pvalue'] = pvalues
                
        df.reset_index(inplace=True)
        
        fig, axs = plt.subplots(3, 2, figsize=(8, 9), sharex=True)
        axs = axs.ravel()

        for stylisedFactNumber, stylisedFact in enumerate(self.stylisedFactsNames):
        
            ax = axs[stylisedFactNumber]
            
            NaNwarning = f'Replaced NaN for {stylisedFact} - Bubble'
            print(NaNwarning) if df[stylisedFact].isnull().values.any() else None
            df[stylisedFact].fillna(0, inplace=True)
                        
            color = df[stylisedFact]
            
            df['MarkerSize'] = df[stylisedFact]-df[stylisedFact].min() #lower bound to 0
            df.MarkerSize = df.MarkerSize/df.MarkerSize.max() #upper bound to 1
            df.MarkerSize -= .5
            df.MarkerSize = df.MarkerSize*2
            df.MarkerSize = 100+ df.MarkerSize.abs()*150
            
            norm = plt.Normalize(df[stylisedFact].min(), df[stylisedFact].max())
            sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
            sm.set_array([])
                        
            graph = sns.scatterplot(data=df, x='alphaConf', y='alphaDisc', ax=ax, c=df[stylisedFact], s=df.MarkerSize, cmap='RdBu_r')
            
            if macrovariableName == 'returns':
            
                #df['pvalue'] = df[stylisedFact].apply(lambda x: stats.anderson_ksamp([ [x], self.dfStylisedFactsLSEReturns[stylisedFact]]).pvalue < 0.05)
                sns.scatterplot(data=df, x='alphaConf', y='alphaDisc', ax=ax, c='None', marker='o', edgecolor='k', linewidth=(1-df[stylisedFact+'_pvalue'])*2, s=df.MarkerSize)
                
                df[stylisedFact]
                self.dfStylisedFactsLSEReturns[stylisedFact].loc[self.LSEstock]
                
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])
            
            xlim = ax.get_xlim()
                   
            ax.plot( [xlim[0], xlim[1]],[xlim[0], xlim[1]], color='k', alpha=.1, linewidth=.5, linestyle='--' )
                       
            ax.set_xlabel(r'$\alpha_{+}$') if stylisedFactNumber >= len(self.stylisedFactsNames)-2 else ax.set_xlabel('')
            ax.set_ylabel(r'$\alpha_{-}$') if np.mod(stylisedFactNumber, 2) == 0 else ax.set_ylabel('')
            #ax.set_yticklabels('') if np.mod(stylisedFactNumber, 2) == 1 else ax.set_ylabel('')
            
            ax.set_title(f'{stylisedFact} -'+ NaNwarning) if df.isnull().values.any() else ax.set_title(stylisedFact)
            
            #ax.get_legend().remove()
            ax.figure.colorbar(sm, ax=ax) #fig.get_axes())
            
        plt.tight_layout()
        plt.savefig(self.plotPathDatasets+f'/Bubbleview_parameter_vs_stylised_facts_{macrovariableName} - REPORT_stats_thr_corrected-ANOVA.pdf', dpi=200)
        #plt.savefig(self.plotPathDatasets+f'/Bubbleview_parameter_vs_stylised_facts_{macrovariableName} - REPORT_stats_thr_corrected-ANOVA_NORMAL_transf.pdf', dpi=200)
        plt.close()
        
############################################################################
        
    def hyperparameters_vs_stylised_facts(self, macrovariableName):
        
        # HP vs SF returns
        #dfhyperparameters = self.dfhyperparameters.loc[self.dfhyperparameters.index.repeat(self.S)].reset_index(drop=True)
        
        df = pd.concat([self.dfStylisedFactsReturns, self.dfhyperparameters], axis=1) if macrovariableName == 'returns' else pd.concat([self.dfStylisedFactsVolumes, self.dfhyperparameters], axis=1)
        
        #breakpoint()
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
                                        
        #    if 'Alpha' in self.parameterName:
        #        x = np.linspace(0, 1, 200)
        #        y = stats.beta.pdf(x, a=hyperparameter, b=1)
                
        #    elif 'alpha' in self.parameterName:
        #        m = self.examplesReady+1 #granularity
        #        x = np.linspace(0, 1, self.examplesReady*m+(m-1))
        #        y = signal.unit_impulse(self.examplesReady*m+(m-1), n*m)
              
            #ax.plot(x, y, color=color, alpha=.6, linewidth=2.3, label=f"{self.hyperparameterName} = {round(hyperparameter, 2)}")
        #    ax.set_xscale('symlog') if 'InitialVolumeOwned' in self.parameterName else None
            
        #ax.set_xlabel(self.parameterName)
        #ax.set_ylabel('Density')
        #plt.legend()

        for stylisedFactNumber, stylisedFact in enumerate(self.stylisedFactsNames):
            ax = axs[3+stylisedFactNumber]
            ax.axhline(y=0, color='k', alpha=.2)

            NaNwarning = f'Replaced NaN for {stylisedFact}'
            print(NaNwarning) if df[stylisedFact].isnull().values.any() else None
            df[stylisedFact].fillna(0, inplace=True)
            
            #breakpoint()
            
            graph = sns.scatterplot(data=df, x='Confirmation bias', y=stylisedFact, ax=ax, alpha=.4, c=df['alphaDisc'], s=df['Confirmation bias']*3+5, cmap=self.cmap)
            sns.scatterplot(data=df.groupby(['alphaConf', 'alphaDisc']).mean(), x='Confirmation bias', y=stylisedFact, ax=ax, color='k', marker='.')
            
            if macrovariableName == 'returns':
                
                graph.axhline(self.dfStylisedFactsLSEReturns.loc[self.LSEstock][stylisedFact], linestyle=':', linewidth=.7, alpha=1, c=[.6, .1, .1], label=self.LSEstock)
                graph.axhline(self.dfStylisedFactsLSEReturns[stylisedFact].mean(), linestyle=':', linewidth=.7, alpha=1, c=[.1, .1, .6], label='LSE mean')
                graph.legend() if stylisedFactNumber == len(self.stylisedFactsNames) - 1 else None
                            
            ax.set_xscale('symlog', base=2) if np.log10(df['Confirmation bias'].max())-np.log10( df['Confirmation bias'][df['Confirmation bias']>0].min() ) >= 2 else ax.yaxis.major.formatter.set_powerlimits((0,0))
            
            ax.set_title(NaNwarning) if df.isnull().values.any() else None
            
        title = '' #f'Hyperparameter of interest: {self.hyperparameterName}'
        #title = title+'- secondary hyperparameter: {self.secondaryHyperparameterName} = {self.secondaryHyperparameter}' if
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.plotPathDatasets+f'/parameter_vs_stylised_facts_{macrovariableName}.png', dpi=200)
        plt.close()
    
    def stylised_facts_distributions_joyplot(self):
    
        from joypy import joyplot
        
        joyplot(df, by='Arm', column='Income', colormap=sns.color_palette("crest", as_cmap=True))
        plt.xlabel('Income')
        plt.title("Ridgeline Plot, multiple groups")
        plt.savefig(stylisedFactsDistributionFolder+f'/{stylisedFact}.png', dpi=200)
        plt.close()
    
    def stylised_facts_distributions(self, macrovariableName):
    
        stylisedFactsDistributionFolder = self.plotPathDatasets+f'/stylised_facts_distributions_{macrovariableName}'
        os.makedirs(stylisedFactsDistributionFolder) if not os.path.isdir(stylisedFactsDistributionFolder) else None #create Data folder for all data files, in case it does not exist
    
        #breakpoint()
        #dfReturns = pd.DataFrame(data=self.StylisedFactsReturns)
    
        #df = pd.concat([dfReturns, self.dfhyperparameters], axis=1) if macrovariableName == 'returns' else pd.concat([self.dfStylisedFactsVolumes, self.dfhyperparameters], axis=1)
        #sns.displot(data=df, y=stylisedFact, hue="Confirmation bias", col="alphaConf", kind="kde", palette='coolwarm', common_grid=True, rug=False, common_norm=True)     
        #title = '' #f'Hyperparameter of interest: {self.hyperparameterName}'
        ##title = title+'- secondary hyperparameter: {self.secondaryHyperparameterName} = {self.secondaryHyperparameter}' if
        #plt.suptitle(title)
        #plt.tight_layout()
        #plt.savefig(stylisedFactsDistributionFolder+f'/{macrovariableName}.png', dpi=200)
        #plt.close()


        # HP vs SF returns        
        df = pd.concat([self.dfStylisedFactsReturns, self.dfhyperparameters], axis=1) if macrovariableName == 'returns' else pd.concat([self.dfStylisedFactsVolumes, self.dfhyperparameters], axis=1)
        
        for stylisedFactNumber, stylisedFact in enumerate(self.stylisedFactsNames):
            sns.displot(data=df, y=stylisedFact, hue="Confirmation bias", col="alphaConf", kind="kde", palette='coolwarm', common_grid=True, rug=False, common_norm=True)
            
            title = '' #f'Hyperparameter of interest: {self.hyperparameterName}'
            #title = title+'- secondary hyperparameter: {self.secondaryHyperparameterName} = {self.secondaryHyperparameter}' if
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(stylisedFactsDistributionFolder+f'/{stylisedFact}.png', dpi=200)
            plt.close()
    
    def hyperparameters_vs_stylised_facts_bubbleview(self, macrovariableName, plot):

        #print(self.dfStylisedFactsReturns)

        # HP vs SF returns        
        df = pd.concat([self.dfStylisedFactsReturns, self.dfhyperparameters], axis=1) if macrovariableName == 'returns' else pd.concat([self.dfStylisedFactsVolumes, self.dfhyperparameters], axis=1)
        #StylisedFactsLSE = self.StylisedFactsLSEreturns if macrovariableName == 'returns' else self.StylisedFactsLSEvolumes
        
        df = df.groupby(['alphaConf', 'alphaDisc']).mean() #take mean across simulations and/or repetitions
        df.reset_index(inplace=True)
        #df['Learning rate difference'] = df['alphaConf']-df['alphaDisc']
                
        fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharex=True)
        axs = axs.ravel()

        gs = GridSpec(3, 3, figure=fig)
        [ax.remove() for ax in axs[:3]] 
        ax = fig.add_subplot(gs[0, :])

        #N = min(4, self.examplesReady-1)
        #for n in np.arange(0, self.examplesReady+1, N): #N examples of distributions
        
        #    hyperparameter = self.hyperparameterRange[n]
        #    color = self.cmap(self.norm(n))
                                        
        #    if 'Alpha' in self.parameterName:
        #        x = np.linspace(0, 1, 200)
        #        y = stats.beta.pdf(x, a=hyperparameter, b=1)
                
        #    elif 'alpha' in self.parameterName:
        #        m = self.examplesReady+1 #granularity
        #        x = np.linspace(0, 1, self.examplesReady*m+(m-1))
        #        y = signal.unit_impulse(self.examplesReady*m+(m-1), n*m)
              
            #ax.plot(x, y, color=color, alpha=.6, linewidth=2.3, label=f"{self.hyperparameterName} = {round(hyperparameter, 2)}")
        #    ax.set_xscale('symlog') if 'InitialVolumeOwned' in self.parameterName else None
            
        #ax.set_xlabel(self.parameterName)
        #ax.set_ylabel('Density')
        #plt.legend()

        for stylisedFactNumber, stylisedFact in enumerate(self.stylisedFactsNames):
        
            ax = axs[3+stylisedFactNumber]
            #ax.axhline(y=0, color='k', alpha=.2)
            
            NaNwarning = f'Replaced NaN for {stylisedFact} - Bubble'
            print(NaNwarning) if df[stylisedFact].isnull().values.any() else None
            df[stylisedFact].fillna(0, inplace=True)
            
            color = df[stylisedFact]
            
            df['MarkerSize'] = df[stylisedFact]-df[stylisedFact].min() #lower bound to 0
            df.MarkerSize = df.MarkerSize/df.MarkerSize.max() #upper bound to 1
            df.MarkerSize -= .5
            df.MarkerSize = df.MarkerSize*2
            df.MarkerSize = 100+ df.MarkerSize.abs()*150
            
            df = df[df.alphaConf<=.01] if plot == 'low' and len(df[df.alphaConf<=.01]) > 0 else df[df.alphaConf>.01] #or df.alphaConf == 0]
            #df = df[df.alphaConf<=.005] if plot == 'low' else df[df.alphaConf>.005 ] #or df.alphaConf == 0]
            
            ax.plot( [0, df.alphaConf.max()],[0, df.alphaConf.max()], color='k', alpha=.1, linewidth=.5, linestyle='--' )
            
            graph = sns.scatterplot(data=df, x='alphaConf', y='alphaDisc', ax=ax, c=df[stylisedFact], s=df.MarkerSize, cmap='RdBu_r') #my_gradient) #c=df['Learning rate difference'], s=(df[stylisedFact]+1)/(df[stylisedFact]+1).min()*5, cmap=self.cmap)
                          
            #ax.set_xlim([-df['alphaConf'][df['alphaConf']>0].min(), df['alphaConf'].max()*1.05])
            #ax.set_ylim([-df['alphaDisc'][df['alphaDisc']>0].min(), df['alphaDisc'].max()*1.05])
            #ax.set_xscale('symlog', base=10) #if np.log10(df['alphaConf'].max()/df['alphaConf'][df['alphaConf']>0].min()) >= 2 else ax.xaxis.major.formatter.set_powerlimits((-3,0))
            #ax.set_yscale('symlog', base=10) #if np.log10(df['alphaDisc'].max()/df['alphaDisc'][df['alphaDisc']>0].min()) >= 2 else ax.yaxis.major.formatter.set_powerlimits((-3,0))
            
            if plot == 'low':
            
                #ax.set_xlim([df['alphaConf'][df['alphaConf']>0].min()*0.75, df['alphaConf'].max()*1.3])
                #ax.set_ylim([df['alphaDisc'][df['alphaDisc']>0].min()*0.75, df['alphaDisc'].max()*1.3])
                ax.set_xlim([-0.001, df['alphaConf'].max()*1.1])
                ax.set_ylim([-0.001, df['alphaDisc'].max()*1.1])
                
            else:
                ax.set_xlim([0, df['alphaConf'].max()*1.1])
                ax.set_ylim([0, df['alphaDisc'].max()*1.1])
            
            #ax.set_xscale('log', base=10) if plot == 'low' else None
            #ax.set_yscale('log', base=10) if plot == 'low' else None
                  
            ax.set_title(f'{stylisedFact} -'+ NaNwarning) if df.isnull().values.any() else ax.set_title(stylisedFact)
            
        title = '' #f'Hyperparameter of interest: {self.hyperparameterName}'
        #title = title+'- secondary hyperparameter: {self.secondaryHyperparameterName} = {self.secondaryHyperparameter}' if
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.plotPathDatasets+f'/Bubbleview_parameter_vs_stylised_facts_{macrovariableName} - {plot}.png', dpi=200)
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
        title = '' #f'Hyperparameter of interest: {self.hyperparameterName}'
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
    
        self.fetch = fetch
        self.InputParameters = fetch.input_parameters()
        self.T = self.InputParameters['T']
        self.Ttraining = self.InputParameters['Ttraining']
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
        
        #Market variables training
        self.PriceTraining = np.zeros((self.Ttraining-tInit))
        self.VolumeTraining = np.zeros((self.Ttraining-tInit))
        self.BidAskSpreadTraining = np.zeros((self.Ttraining-tInit))
        
        #Market variables testing        
        self.Price = np.zeros((self.S, self.T-tInit))
        self.Volume = np.zeros((self.S, self.T-tInit))
        self.BidAskSpread = np.zeros((self.S, self.T-tInit))
        
        marketParameters = self.fetch.parameters('market')
        self.tick = marketParameters['tick']
        
        #RL variables
        #self.QT = np.zeros((self.S, self.I, self.T-tInit+1, 3, 2, 2, 3, 3))
        #self.QCA = np.zeros((self.S, self.I, self.T-tInit+1, 3, 2, 2, 2))
        #self.QCB = np.zeros((self.S, self.I, self.T-tInit+1, 3, 2, 2, 2))
        
        ### Load values
                
        ### Testing
        for s in range(self.S):
        
            self.Price[s] = fetch.saved('Price', s=s, verbose=False)[0, tInit:]
            self.Volume[s] = fetch.saved('Volume', s=s, verbose=False)[0, tInit:]
            self.BidAskSpread[s] = fetch.saved('BidAskSpread', s=s, verbose=False)[0, tInit:]
            
            #self.QT[s] = fetch.saved('QT', s=s, verbose=False)[:, tInit-1:]
            #self.QCA[s] = fetch.saved('QCA', s=s, verbose=False)[:, tInit-1:]
            #self.QCB[s] = fetch.saved('QCB', s=s, verbose=False)[:, tInit-1:]
            
        self.returns = np.log10(self.Price[:, 1:]) - np.log10(self.Price[:, :-1]) #returns
        self.rdf = pd.DataFrame(data=self.returns.T, columns=list(range(self.S)))
        
        self.vdf = pd.DataFrame(data=self.Volume.T, columns=list(range(self.S)))
        
        #self.QF_dynamics()
        #self.QT_dynamics( s )
        #self.QC_dynamics('Asks')
        #self.QC_dynamics('Bids')
        
    def market_dynamics(self, examplePlotFolder, phase = 'training'):
    
        self.create_example_plot_folders(examplePlotFolder)
    
        if phase == 'training':

            Price = self.fetch.saved('Price', verbose=False)[0, tInit:]
            Volume = self.fetch.saved('Volume', verbose=False)[0, tInit:]
            BidAskSpread = self.fetch.saved('BidAskSpread', verbose=False)[0, tInit:]  
            returns = np.log10(Price[1:]) - np.log10(Price[:-1]) #returns    	
    	
        else:
    	
            Price = self.Price
            Volume = self.Volume
            BidAskSpread = self.BidAskSpread 
            returns = np.log10(Price[:, 1:]) - np.log10(Price[:, :-1]) #returns  

        for j in range(self.J):

            plt.figure(figsize=(10, 7))

            #Price dynamics
            ax1 = plt.subplot(411)
            
            if phase == 'training':
                ax1.plot(Price, color='k')
            else:
                [ax1.plot(Price[s], alpha=.5) for s in range(self.S)]
           
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
            ax2 = plt.subplot(412, sharex=ax1)
            ax2.plot(self.Volume, color='k')
            
            if phase == 'training':
                ax2.plot(Volume, color='k')
            else:
                [ax2.plot(Volume[s], alpha=.5) for s in range(self.S)]
            
            #ax2.set_xlabel('Time step')
            ax2.set_ylabel('Traded volume')
            ax2.set_yscale('symlog')
            plt.tick_params('x', labelbottom=False)

            #Bid-ask spread
            ax3 = plt.subplot(413, sharex=ax1)
            ax3.plot(BidAskSpread, color='k', alpha=.5)
            
            if phase == 'training':
                ax3.plot(BidAskSpread, color='k')
            else:
                [ax3.plot(BidAskSpread[s], alpha=.5) for s in range(self.S)]
            
            
            #ax3.set_xlabel('Time step')
            ax3.set_ylabel('Bid-Ask spread (€)')
            ax3.set_ylim([self.tick/2, None])
            ax3.set_yscale('symlog') #, linear_width=tick, subs=[-2, -1, 0, 1])
            plt.tick_params('x', labelbottom=False)
            #plt.savefig(self.plotPath+f'/Trading_volume_dynamics_stock{j}', dpi=120)

            ax4 = plt.subplot(414, sharex=ax1)
            sliding_window_view = np.lib.stride_tricks.sliding_window_view

            if phase == 'training':
                ax4.plot(sliding_window_view(returns, window_shape = 5).var(axis=1), color='k')
            else:
                [ax4.plot(sliding_window_view(returns[s], window_shape = 5).var(axis=1), alpha=.5) for s in range(self.S)]

            ax4.set_xlabel('Time step (trading days)')
            ax4.set_ylabel('Volatility ( Var[r] )')
            ax4.set_yscale('symlog')
            #plt.savefig(self.plotPath+f'/Volatility_stock{j}', dpi=120)

            #ax.spines['top'].set_visible(False)
            #ax.spines['right'].set_visible(False)

            plt.savefig(examplePlotFolder+f'/Market_dynamics_{phase}_stock{j}', dpi=120)            

            plt.close()
            
    def create_example_plot_folders(self, examplePlotFolder):
        
        os.makedirs(examplePlotFolder) if not os.path.isdir(examplePlotFolder) else None #create Data folder for all data files, in case it does not exist
    
    def attempts_vs_wealth(self, examplePlotFolder):
  
        actionT = self.fetch.saved('actionT')[:, :, tInit:]
        Atotal = self.Atotal = self.fetch.saved('Atotal')[:, tInit-1:]
        VolumeOwned = self.fetch.saved('VolumeOwned')[:, :, tInit-1:]
        
        self.create_example_plot_folders(examplePlotFolder)
        
        cmap = 'turbo'

        fig = plt.figure(figsize=(9, 9))

        x = (abs(actionT[:, 0, :-1, 0]) > 0).sum(axis=1)
        y = Atotal[:, -1]
        s = np.log(VolumeOwned[:, 0, -1]+10)*10
        color = (abs(VolumeOwned[:, 0, 1:]-VolumeOwned[:, 0, :-1]) > 0).sum(axis=1)

        plt.scatter(x, y, s=s, alpha=.4, c=color, cmap=cmap)
        plt.title('size = stocks owned, color = resolved trading')
        plt.xscale('symlog', base=2)
        plt.yscale('symlog', base=10)
        plt.xlabel('Trading actions taken')
        plt.ylabel('Total final wealth ($)')
        
        #breakpoint()
        plt.savefig(examplePlotFolder+'/Attempts_vs_wealth', dpi=120)
        plt.close()
        
    def QF_dynamics(self, examplePlotFolder):
    
        self.QFtraining = np.zeros((self.I, self.Ttraining-tInit+1, 3))
        self.QFtraining = self.fetch.saved('QF', s='training', verbose=False)[:, tInit-1:]
        
        self.QF = np.zeros((self.S, self.I, self.T-tInit+1, 3))
        
        for s in range(self.S):
            self.QF[s] = self.fetch.saved('QF', s=s, verbose=False)[:, tInit-1:]
        
        self.create_example_plot_folders(examplePlotFolder)
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 9), sharex=True, sharey=True)
            
        for forecastStrategy in range(3):
            ax = axs[forecastStrategy]
            ax.plot(self.QFtraining[:, :, forecastStrategy].mean(axis=0).T, linewidth=2, color='k')
            ax.plot(self.QFtraining[:, :, forecastStrategy].T)
            
        title = f'Training' #f'Hyperparameter of interest: {self.hyperparameterName}'
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(examplePlotFolder+f'/Training - QF.png', dpi=200)
        plt.close()
            
        ###
            
        fig, axs = plt.subplots(1, 3, figsize=(12, 9), sharex=True, sharey=True)
            
        for forecastStrategy in range(3):
            ax = axs[forecastStrategy]
            ax.plot(self.QF[:, :, :, forecastStrategy].mean(axis=0).T)
            
        title = f'Mean' #f'Hyperparameter of interest: {self.hyperparameterName}'
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(examplePlotFolder+f'/Mean over sim - QF.png', dpi=200)
        plt.close()
            
        ###
            
        #for s in range(self.S):
            
        #    fig, axs = plt.subplots(1, 3, figsize=(9, 9), sharex=True, sharey=True)
        #    axs = axs.ravel()
        
        #    for forecastStrategy in range(3):
        #        ax = axs[forecastStrategy]
        #        ax.plot(self.QF[s, :, :, forecastStrategy].T)
        
        #    title = f'Sim: {s}' #f'Hyperparameter of interest: {self.hyperparameterName}'
        #    plt.suptitle(title)
        #    plt.tight_layout()
        #    plt.savefig(plotPathDatasets+f'/Sim. {s} QF.png', dpi=200)
        #    plt.close()
        
        ###

        colours = ['k', 'r', 'b']

        fig, axs = plt.subplots(10, 10, figsize=(9, 9), sharex=True, sharey=True)
        axs = axs.ravel()
        
        [axs[i].plot(self.QFtraining[i], alpha=.4) for i in range(self.I)]
        
        [[line.set_color(colours[idx]) for idx, line in enumerate(axs[i].lines)] for i in range(self.I)]
                
        title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(examplePlotFolder+f'/Trader training - QF.png', dpi=200)
        plt.close()

        ###
        
        fig, axs = plt.subplots(10, 10, figsize=(9, 9), sharex=True, sharey=True)
        axs = axs.ravel()
        
        [axs[i].plot(self.QF[:, i].mean(axis=0), alpha=.4) for i in range(self.I)]
        [[line.set_color(colours[idx]) for idx, line in enumerate(axs[i].lines)] for i in range(self.I) ]
            
        title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(examplePlotFolder+f'/Trader means over sims - QF.png', dpi=200)
        plt.close()
        
        return
        
    
            
        #def QT_dynamics(self):
        
        #    return

        #def QC_dynamics(self, OrderType):
        
        #    return
        
