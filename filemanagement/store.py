#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 17:39:40 2022

@author: svrizzi
"""

import os
import numpy as np
import json
from numpyencoder import NumpyEncoder
import pandas as pd
import pickle

class FileStorage:
    
    def __init__(self, datasetFolder=None, verbose=True, calibrating=False):

        self.verbose = verbose
        self.calibrating = calibrating
        self.OutputFolder = 'output'
        self.DataFolder = 'data'
        self.PlotsFolder = 'plots'
        self.ReportFolder = 'reports'
        self.EmpiricalParametersFolder = 'Empirical RL Parameters'
        self.FileExtension = 'json' #pickle
        self.write = 'w' #wb
                
        #Output folder paths
        self.DataFolderPath = f'{self.OutputFolder}/{self.DataFolder}' if not datasetFolder else f'{self.OutputFolder}/{self.DataFolder}/'+datasetFolder
        self.ReportFolderPath = f'{self.OutputFolder}/{self.ReportFolder}'
        self.EmpiricalParametersFolderPath = f'{self.OutputFolder}/{self.EmpiricalParametersFolder}'

        if os.path.basename(os.path.normpath(os.getcwd())) == 'analysis': #if you are in the analysis folder

            self.DataFolderPath = '../'+self.DataFolderPath
            self.EmpiricalParametersFolderPath = '../'+self.EmpiricalParametersFolderPath

        #Filenames
        self.masterFilename = 'master'
        self.trainingDataFilename = 'hyperparameters' #'tweaked_hyperparameters'
        self.EmpiricalParametersFile = 'df_par_pool_False.csv'
        
        #Files
        self.masterFile = f'/{self.masterFilename}.{self.FileExtension}'
        self.masterTimeFile = f'/{self.masterFilename}_time.{self.FileExtension}'
                
        self.parameterFileNames = ['RL', 'SYMBA', 'wallet', 'market']
                	
        self.parameterFiles = [self.parameterfile_naming(parameterFilename) for parameterFilename in self.parameterFileNames]
        self.RLParameterFile = self.parameterFiles[0]
        self.SYMBAParameterFile = self.parameterFiles[1]
        self.walletParameterFile = self.parameterFiles[2]
        self.marketParameterFile = self.parameterFiles[3]
        
        self.trainingDataHyperparameterFile = f'/{self.trainingDataFilename}.{self.FileExtension}'
        
        self.exampleSet = False
        
        #Create new folder for current batch of simulations
        #adapted from https://stackoverflow.com/questions/63261580/python-making-a-folder-with-incrementing-number-depending-on-existing-folders-w
        
        self.flagFilename=self.DataFolderPath+'/flag.npy'
        
        self.plotFolderPath = f'{self.OutputFolder}/{self.PlotsFolder}'
     
        self.MarketVariables = ['Price', 'Volume', 'BidAskSpread', 'fundamentalValueTrue']
        self.LearningVariables = ['QForecasting', 'VForecasting', 'QTrading', 'VTrading', 'QPricing', 'VPricing', 'QReferencing', 'VReferencing']
        self.StockDependentLearningVariables = ['VolumeOwned', 'Aequity', 'AvailableVolume',  'stateForecasting', 'actionForecasting', 'stateTrading', 'actionTrading', 'actionTradingChosen', 'actionTradingEffective', 'statePricing', 'actionPricing', 'Forecast', 'averageBuyPrice', 'fundamentalValue', 'BalanceSheet', 'knowledgeWaste']
        self.WealthVariables = ['Wallet', 'AvailableMargin', 'Atotal', 'BalanceSheetBank']

        self.dfMicroMergedName = 'df_micro_merged.csv'        
        self.dfMicroCoefName = 'df_micro_coef.csv'
        self.regressorDictionaryName = 'regressors_dict'
        
###################### Set folder and paths to example of interest ###########################
    
    def output_folders(self):

        #Create folders
        os.mkdir(self.OutputFolder) if not os.path.isdir(self.OutputFolder) else None #create Data folder for all data files, in case it does not exist
        os.mkdir(self.DataFolderPath) if not os.path.isdir(self.DataFolderPath) else None #create Data folder for all data files, in case it does not exist
        #os.mkdir(self.ReportFolderPath) if os.path.isdir(self.ReportFolderPath) == False else None #create Report folder for all reports, in case it does not exist
        os.mkdir(self.EmpiricalParametersFolderPath) if not os.path.isdir(self.EmpiricalParametersFolderPath) else None

    def set_example(self, exampleNumber=-1):
        
        self.FlagSet = False
        self.exampleNumber = exampleNumber
        self.exampleSet = True
        
    def example_folder_path_name(self):

        return f'/ex{self.flag}'

    def check_flag(self): #not used
    
        print(self.flagFilename)
        
        try:
            flag = np.load(self.flagFilename) #allow_pickle=True
            
        except FileNotFoundError:
            
            flag=0

        return flag

    def set_flag(self, exampleNumber=-1):

        try:

            flag = np.load(self.flagFilename) #allow_pickle=True
            flag = int(flag) + 1
                           
        except FileNotFoundError:
            
            flag=0
            
        self.flag = flag if exampleNumber == -1 else exampleNumber

        np.save(self.flagFilename, self.flag)
        self.FlagSet = True

    def new_example(self, exampleNumber):

        if exampleNumber == -1:
        
            self.set_flag(exampleNumber)
        
        else:
        
            self.flag = exampleNumber
            self.FlagSet = True

        print('Creating...'+self.example_folder_path_name())
        
        path = self.DataFolderPath+self.example_folder_path_name()

        os.makedirs(path, exist_ok=True) #if not os.path.isdir(path) else None
        
    def example_folder_path(self):
        
        if self.FlagSet:
            
            pass

        else:          

            if self.exampleNumber >= 0: #desidered example number
            
                self.flag = self.exampleNumber
            
            elif self.exampleNumber == -1: #latest example
                
                try:

                    self.flag = int(np.load(self.flagFilename))
                    
                except FileNotFoundError:
                    
                    self.flag = int(np.load('../'+self.flagFilename))
                    
        return self.example_folder_path_name()
    
    def simulation_folder_path(self, s):
        
        self.SimulationPath = self.DataFolderPath+self.example_folder_path()+f'/sim_{s}'
        
        os.mkdir(self.SimulationPath) if not os.path.isdir(self.SimulationPath) else None #create Data folder for all data files, in case it does not exist

        return self.SimulationPath
    
    def t_init(self, tInit):
        self.tInit = tInit

###################### Functions to save files ###########################

    def master_file(self, data, time=False):
           
        print('Saving master file...') if self.verbose else None
            
        #data = dict(zip(kwargs.keys(), kwargs.values()))
            
        FileName = self.DataFolderPath+self.example_folder_path()
        FileName += self.masterFile if not time else self.masterTimeFile
        
        with open(FileName, self.write) as f: # An arbitrary collection of objects supported by pickle.
            
            json.dump(data, f, cls=NumpyEncoder) # Pickle the 'data' dictionary using the highest protocol available
            
    def parameterfile_naming(self, parameterFilename):
        return f'/{parameterFilename}_parameters.{self.FileExtension}'
        
    def parameters(self, parameters, parameterName):
           
        print(f'Saving {parameterName} parameters...') if self.verbose else None
        
        if parameterName in self.parameterFileNames:
        
            ParameterFile = self.parameterfile_naming(parameterName)
            FilePath = self.DataFolderPath+self.example_folder_path()+ParameterFile
            FilePath += '_calibrating' if self.calibrating else ''

        else: #elif parameterName == 'training_data':

            FilePath = self.DataFolderPath+self.trainingDataHyperparameterFile
    
        with open(FilePath, self.write) as f: # An arbitrary collection of objects supported by pickle.
          
            json.dump(parameters, f, cls=NumpyEncoder, indent=0) #

    def sensitivity(self, exStart):
           
        print(f'Saving sensitivity master file...') if self.verbose else None
        
        keys = ['exStart']
        values = [exStart]
        comparisonDict = dict( zip(keys, values) )
        
        prefixed = [filename for filename in os.listdir(self.DataFolderPath) if filename.startswith("sensitivity")]
        n = max( [int(filename.split('_')[-1].split('.json')[0]) for filename in prefixed] ) + 1 if len(prefixed) > 0 else 0
        File = f'/sensitivity_{n}.json'
        FilePath = self.DataFolderPath+File
        #FilePath += ''

        #FilePath = self.DataFolderPath+self.trainingDataHyperparameterFile
    
        with open(FilePath, self.write) as f: # An arbitrary collection of objects supported by pickle.
          
            json.dump(comparisonDict, f, cls=NumpyEncoder, indent=0) #

    def comparison(self, randomSeeds, cognitiveTraits, exStart):
           
        print(f'Saving comparison master file...') if self.verbose else None
        
        keys = ['randomSeeds', 'cognitiveTraits', 'exStart']
        values = [randomSeeds, cognitiveTraits, exStart]
        comparisonDict = dict( zip(keys, values) )
        
        prefixed = [filename for filename in os.listdir(self.DataFolderPath) if filename.startswith("comparison")]
        n = max( [int(filename.split('_')[-1].split('.json')[0]) for filename in prefixed] ) + 1 if len(prefixed) > 0 else 0
        File = f'/comparison_{n}.json'
        FilePath = self.DataFolderPath+File
        #FilePath += ''

        #FilePath = self.DataFolderPath+self.trainingDataHyperparameterFile
    
        with open(FilePath, self.write) as f: # An arbitrary collection of objects supported by pickle.
          
            json.dump(comparisonDict, f, cls=NumpyEncoder, indent=0) #

    def empirical_RL_parameters(self, dfPar):
    
        dfPar.to_csv(f'{self.EmpiricalParametersFolderPath}/{self.EmpiricalParametersFile}', index=False) #get parameter data file from folder or github, if does not exist 
    
    def data_arrays_file(self):
    
        return '/data_arrays' #f'data_arrays.{self.FileExtension}'
    
    def data_lists_file(self):
    
        return f'/data_lists.{self.FileExtension}'
    
    def data(self, variables, s, T, I):
        
        print(f'Saving data simulation {s}... \n') if self.verbose else None
        
        Arrays = {}
        Lists = {}
        
        for key, value in variables.__dict__.items(): #loop over variables
            
            if isinstance(value, np.ndarray):
                
                #value = value[:t] if value.shape[0] == T else value
                Arrays[key] = value
                
                #np.save(key, value)
                
                #value = np.reshape(value, (T, I)) if np.size(value) == T * I else value
                #try:
                #    df = pd.DataFrame(data=value)
                #except:
                #    breakpoint()
                #df.to_csv(self.simulation_folder_path(s)+f'/{key}.csv')


            elif isinstance(value, list):
                
                if any(isinstance(item, pd.DataFrame) for item in value): #DataFrames (Order Books)
                    [item.to_csv(self.simulation_folder_path(s)+f'/{key}_stock_{j}.csv') for j, item in enumerate(value)]

                else:
                    Lists[key] = value
            
        FilenameArrays = self.simulation_folder_path(s)+self.data_arrays_file()
        #with open(FilenameArrays, self.write) as f: # An arbitrary collection of objects supported by pickle.
        #    json.dump(Arrays, f, cls=NumpyEncoder)
            
        np.savez_compressed(FilenameArrays, **Arrays)

        FilenameLists = self.simulation_folder_path(s)+self.data_lists_file()
        with open(FilenameLists, self.write) as f: # An arbitrary collection of objects supported by pickle.
            json.dump(Lists, f, cls=NumpyEncoder)


    def data3(self, variables, s='training'):
              
        sTemp = s.split('_')[0] if isinstance(s, str) else 'testing'
        
        Arrays = {}
        
        for key, value in variables.market.__dict__.items(): #loop over variables

            if isinstance(value, np.ndarray):
            
                if sTemp in ['pretraining']:
                                
                    if key in self.LearningVariables:
                        
                        Arrays[key] = value
                        #Arrays[key+'_end'] = value[:, -self.tInit:]

                    elif key in ['priceDictionary', 'volatilityReference', 'BidAskSpreadReference']:
                        Arrays[key] = value
                                    
                elif sTemp in ['training', 'testing']:
                
                    if key in self.LearningVariables+self.WealthVariables+self.MarketVariables:

                        Arrays[key] = value
                        #Arrays[key+'_end'] = value[:, -self.tInit:]
                    
                    elif key in self.StockDependentLearningVariables: #+['RewardTradingFactual', 'RewardTrading']:
                        
                        Arrays[key] = value #if key is not 'fundamentalValue' else value[:, :, -1]
                        #Arrays[key+'_end'] = value[:, :, -self.tInit:]
                        
                    elif key in ['priceDictionary', 'volatilityReference', 'BidAskSpreadReference']:
                        Arrays[key] = value
            
            else:

                if key in ['volatilities']:
                    Arrays[key] = value
            
        FilenameArrays = self.simulation_folder_path(s)+self.data_arrays_file()
        np.savez_compressed(FilenameArrays, **Arrays)
        
    def order_books(self, example, s='training'):
    
        print('Save order books...') if self.verbose else None

        [OrderBookAsks.to_csv(self.simulation_folder_path(s)+f'/OrderBookAsks_{stock}.csv') for (stock, OrderBookAsks) in enumerate(example.market.OrderBookAs)]
        [OrderBookBids.to_csv(self.simulation_folder_path(s)+f'/OrderBookBids_{stock}.csv') for (stock, OrderBookBids) in enumerate(example.market.OrderBookBs)]
          

    def df_micro_coeff(self, df, folder):
    
        df.to_csv(folder+f'/{self.dfMicroCoefName}', index=False) #get parameter data file from folder or github, if does not exist 
        
    def df_micro_merged(self, df, folder):
    
        df.to_csv(folder+f'/{self.dfMicroMergedName}', index=False) #get parameter data file from folder or github, if does not exist 

    def regressor_dict(self, dictionary, folder):
    
        with open(folder+f'/{self.regressorDictionaryName}', "wb") as fp:   #Pickling
            pickle.dump(dictionary, fp)    

#%%

#OrderBooksFolder = 'Order books'
#OrderBooksFolderPath = DataFolderPath+NewfolderPath+f'/{OrderBooksFolder}'
#os.mkdir(OrderBooksFolderPath) if os.path.isdir(DataFolderPath+NewfolderPath+OrderBooksFolder) == False else None #create Report folder for all reports, in case it does not exist

#def order_book_asks_filename(j):

#    return OrderBooksFolderPath+f'/OBA{j}.csv'

#def order_book_bids_filename(j):

#    return OrderBooksFolderPath+f'/OBB{j}.csv'

#def save_order_book_asks(df, j):
    
#    df.to_csv(OrderBooksFolderPath+f'/OBA{j}.csv', index=False)
    
#def save_order_book_bids(df, j):
    
#    df.to_csv(OrderBooksFolderPath+f'/OBB{j}.csv', index=False)

### old data3

        #for key, value in variables.market.__dict__.items(): #loop over variables
        #    
        #    if isinstance(value, np.ndarray) and key in self.listEssentialVariables:
            
        #        Arrays[key] = value

        #        if sTemp == 'pretraining':
                                
        #            if key in self.LearningVariables:
        #                Arrays[key+'_end'] = value[:, -tInit:]
                                    
        #        elif sTemp == 'training':
                
        #            if key in self.LearningVariables+['RewardTradingFactual']+self.MarketVariables+self.WealthVariables:
        #                Arrays[key+'_end'] = value[:, -tInit:]
                    
        #            elif key in self.StockDependentLearningVariables:
        #                Arrays[key+'_end'] = value[:, :, -tInit:]
