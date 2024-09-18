#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:27:55 2022

@author: svrizzi
"""

import pandas as pd
import os
import json
import numpy as np

import pickle
import requests
import urllib.request
from io import StringIO

class FileFetching:
    
    def __init__(self, store):
        
        self.store = store
        self.verbose = store.verbose
        self.calibrating = store.calibrating
        
        self.DataFolderPath = store.DataFolderPath
        self.DataFolderPath = self.DataFolderPath if os.path.isdir(store.DataFolderPath) else '../'+self.DataFolderPath
    
        if store.exampleSet:
            
            self.exampleNumber = store.exampleNumber
            
            #if self.exampleNumber >=0:
            
            #    try:
            #        self.flag = int(np.load(store.flagFilename))
            #    except FileNotFoundError:
            #        self.flag = int(np.load('../'+store.flagFilename))
            
            #elif self.exampleNumber == -1:
                
            #    self.flag = self.exampleNumber
            
            self.example_folder_path = store.example_folder_path
            
        self.EmpiricalParametersFolderPath = store.EmpiricalParametersFolderPath
        #from datamanagement.store_data import order_book_asks_filename, order_book_bids_filename
        self.masterFile = store.masterFile
        self.masterTimeFile = store.masterTimeFile
        
        self.parameterfile_naming = store.parameterfile_naming
        self.parameterFileNames = store.parameterFileNames
        self.RLParameterFile = store.RLParameterFile
        self.SYMBAParameterFile = store.SYMBAParameterFile
        self.walletParameterFile = store.walletParameterFile
        self.marketParameterFile = store.marketParameterFile
        self.EmpiricalParametersFile = store.EmpiricalParametersFile
        self.empirical_RL_parameters = store.empirical_RL_parameters
        self.simulation_folder_path = store.simulation_folder_path
        self.data_arrays_file = store.data_arrays_file
        self.flagFilename = store.flagFilename   
        
        self.trainingDataHyperparameterFile = store.trainingDataHyperparameterFile
        
        self.plotFolderPath = store.plotFolderPath

        self.dfMicroMergedName = store.dfMicroMergedName        
        self.dfMicroCoefName = store.dfMicroCoefName
        self.regressorDictionaryName = store.regressorDictionaryName

    def RL_parameters_data(self, model):
        
        '''
        Fetch dataframe with fitted parameter values
        '''

        FolderPath = f'{self.EmpiricalParametersFolderPath}/{self.EmpiricalParametersFile}'
        
        FileSource = 'folder' if os.path.exists(FolderPath) else 'Github'
        #breakpoint()

        print('Empirical RL parameters not found locally...') if FileSource == 'Github' and self.verbose else None
        print(f'Fetching empirical RL parameters from {FileSource}...') if self.verbose else None
        
        try:
        
            dfPar = pd.read_csv(FolderPath)
            #breakpoint()
        
        except FileNotFoundError: #take data from private Github repo
            
            try:
                urllib.request.urlopen('http://github.com') #Python 3.x
                #return True
            except:
                print("\n ** Make sure you have internet connection ** \n")
                
            #adapted from https://stackoverflow.com/questions/62164400/how-to-access-private-github-repo-file-csv-in-python-using-pandas-or-requests
            # define parameters for a request
            token = ''
            owner = 'stefanovrizzi'
            repo = 'hrl-test-retest-reliability'
            path = self.EmpiricalParametersFile
            
            # send a request
            r = requests.get('https://api.github.com/repos/{owner}/{repo}/contents/{path}'.format(owner=owner, repo=repo, path=path),
                             headers={'accept': 'application/vnd.github.v3.raw',
                                      'authorization': 'token {}'.format(token)})
        
            string_io_obj = StringIO(r.text) # convert string to StringIO object
            dfPar = pd.read_csv(string_io_obj)
            
            #url = 'https://github.com/stefanovrizzi/hrl-test-retest-reliability/blob/main/df_par_pool_False.csv'
            #df_par = pd.read_csv(url)
            self.empirical_RL_parameters(dfPar)

        #breakpoint()
    
        # Work on dataframe
        dfPar.set_index(['model', 'exp', 'id', 'parameter'], inplace=True)
        dfSliced = dfPar.xs(model, level='model').unstack(['exp', 'parameter']).xs('Fit', level=None, axis=1) #slice model of interest, stack dataframe and slice column with fitted parameter values 
        dfSliced = dfSliced [[parameter for parameter in dfSliced if len(set(dfSliced[parameter]))>1]] #drop parameters not belonging to model (they have the same fixed value 10**(-7))
        
        ### Test that test-retest distributions do not differ
    
        #from scipy.stats import ttest_ind
        #[print(ttest_ind(dfSliced[0, parameter], dfSliced[1, parameter])[1]) for parameter in df.columns.get_level_values('parameter').unique()]
        #[print(ttest_ind(dfSliced[0, parameter], dfSliced[1, parameter])[1]<0.01) for parameter in df.columns.get_level_values('parameter').unique()]
    
        # Get parameter values
        dfSliced = dfSliced.stack('exp')
        betaDistribution = dfSliced['beta'].values.round(3)
        alphaConfDistribution = dfSliced['alpha_conf'].values.round(3)
        alphaDiscDistribution = dfSliced['alpha_disc'].values.round(3)
        
        return betaDistribution, alphaConfDistribution, alphaDiscDistribution

    def time_references(self):
    
        f = open('initialise/time_parameters.json', 'r')
        data = json.load(f)
        Tw = data['Tw']
        Tm = data['Tm']
        Ty = data['Ty']
        f.close() # Closing file
        
        return Tw, Tm, Ty
    
    def input_parameters(self, time=False):

        filename = self.DataFolderPath+self.example_folder_path()
        filename += self.masterFile if not time else self.masterTimeFile

        #try:
        f = open(filename,) # Opening JSON file    
        #except:
            #f = open('../'+self.DataFolderPath+self.example_folder_path()+self.masterFile,) # Opening JSON file

        data = json.load(f) # returns JSON object as a dictionary
        f.close() # Closing file
        
        return data
    
    def parameters_json(self, parameterName):
        
        if parameterName in self.parameterFileNames:
        
            ParameterFile = self.parameterfile_naming(parameterName)
            FilePath = self.DataFolderPath+self.example_folder_path()+ParameterFile
            FilePath += '_calibrating' if self.calibrating else ''

        else: #elif parameterName == 'training_data':

            FilePath = self.DataFolderPath+self.trainingDataHyperparameterFile
        
        f = open(FilePath, 'r')
        data = json.load(f)
        f.close() # Closing file
    
        return data #parametersRetuned, data.keys()

    def parameters_csv(self, parameterName):
        
        if parameterName in self.parameterFileNames:
        
            FilePath = self.DataFolderPath+self.example_folder_path()+f'/{parameterName}_parameters'
            FilePath += '_calibrating' if self.calibrating else ''

        else: #elif parameterName == 'training_data':

            FilePath = self.DataFolderPath+self.trainingDataHyperparameterFile
        
        df = pd.read_csv(FilePath+'.csv')
        
        return df

    def comparison(self, n):
        
        prefixed = [filename for filename in os.listdir(self.DataFolderPath) if filename.startswith("comparison")]
        #breakpoint()
        if n == -1:
            n = prefixed[-1].split('_')[-1].split('.json')[0] if len(prefixed) > 0 else 0
        else:
            n
           
        File = f'/comparison_{n}.json'
        FilePath = self.DataFolderPath+File
        
        with open(FilePath) as f:
          
            data = json.load(f)

        return data, n #parametersRetuned, data.keys()

    def df_micro_merged(self, folder):
            
        try:
            df = pd.read_csv(folder+f'/{self.dfMicroMergedName}')
            #df.set_index('index', inplace=True)
                             
        except:
            df = None

        return df

    def df_micro_coef(self, folder):
        
        try:
            df = pd.read_csv(folder+f'/{self.dfMicroCoefName}')
            df.set_index('index', inplace=True)
            
            with open(folder+f'/{self.regressorDictionaryName}', "rb") as fp:   # Unpickling
                dictionary = pickle.load(fp)
                 
        except:
            df = None
            dictionary = None
            
        return df, dictionary

    def set_folder_name(self, phase, s):
        
        inputParameters = self.input_parameters(time=True)
        pretrainingFlag = inputParameters['pretrainingFlag']
        
        if phase=='pretraining' and s == 0:
           
           self.FolderName = None
           
        elif phase=='training' and s == 0:
           
           if pretrainingFlag:
           
              S  = inputParameters['Spretraining']
              self.FolderName = f'pretraining_{S-1}'
           
           elif not pretrainingFlag:
              
              self.FolderName = None
              
        elif phase in ['pretraining', 'training'] and s > 0:

            self.FolderName = f'{phase}_{s-1}'

        elif phase=='testing':
        
              S  = inputParameters['Straining']
              self.FolderName = f'training_{S-1}'
       
    def saved_data(self, s='training'):
        
        data = {}
        
        print('Loading from folder '+self.simulation_folder_path(s)+'...') if self.verbose else None
        f = open(self.simulation_folder_path(s)+self.data_arrays_file()+'.npz',) # Opening JSON file
        data = json.load(f) # returns JSON object as a dictionary
        f.close() # Closing file
        
        return data
    
    def saved(self, x, s=None, verbose=False):

        s = s if s else self.FolderName
 
        print(f'Loading {x} from folder '+self.simulation_folder_path(s)+'...') if verbose == True else None

        data = np.load(self.simulation_folder_path(s)+self.data_arrays_file()+'.npz') #allow_pickle=True
        
        return data[x]
    
    def order_books(self, OrderBookAs, OrderBookBs, stocks, s):
    
        for stock in stocks:
        
            OrderBookAsks = pd.read_csv(self.simulation_folder_path(s)+f'/OrderBookAsks_{stock}.csv')
            AskOrderBookName = OrderBookAsks.columns[0]
            OrderBookAsks.index.rename(AskOrderBookName, inplace=True)
            OrderBookAsks.drop(AskOrderBookName, axis=1, inplace=True)
            OrderBookAs[stock] = OrderBookAsks
        
            OrderBookBids = pd.read_csv(self.simulation_folder_path(s)+f'/OrderBookBids_{stock}.csv')
            BidOrderBookName = OrderBookBids.columns[0]
            OrderBookBids.index.rename(BidOrderBookName, inplace=True)
            OrderBookBids.drop(BidOrderBookName, axis=1, inplace=True)
            OrderBookBs[stock] = OrderBookBids
            
        return OrderBookAs, OrderBookBs
    
    def all_dataset_folders(self, prefix=None):

        if not prefix:
        
            directoriesList = os.listdir(self.DataFolderPath)
            directoriesList = [directory for directory in directoriesList if not directory.startswith('ex')]
            directoriesList = [directory for directory in directoriesList if not directory.startswith('flag')]
        
        elif prefix:
        
            directoriesList = [directory for directory in os.listdir(self.DataFolderPath) if directory.startswith(prefix)]

        return directoriesList
        
        

#%%

#def fetch_order_book_asks(j):
    
#    return pd.read_csv(order_book_asks_filename(j), index_col=False)
    
#def fetch_order_book_bids(j):
    
#    return pd.read_csv(order_book_bids_filename(j), index_col=False)

