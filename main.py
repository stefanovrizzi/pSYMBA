#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:20:29 2022

@author: svrizzi
"""

################ Import key functions ############################

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching
import json
from initialise.sample_parameters import ParameterSampling

from simulate.simulation import pSYMBA

import argparse

#from simulation_analysis import SimulationAnalysis
#from report import WriteReport

import time
import numpy as np
import random

################ Parallel computation ############################

import multiprocessing as mp

#%%

#python3 -c "import main; print(pSYMBA)"

class Main:
    
    def __init__(self, hyperparameters_to_tweak={}, CognitiveTraits=None, randomSeed=None, exampleNumber=-1, verbose=False):
        
        self.verbose=verbose
        
        self.setup()
        
        self.get_parser()
                
        self.args.CognitiveTraits = CognitiveTraits if CognitiveTraits else self.args.CognitiveTraits
        #self.args.CognitiveTraits+['no bias'] if 'no bias' not in self.args.CognitiveTraits and 'pessimistic bias' not in self.args.CognitiveTraits and 'optimistic bias' not in self.args.CognitiveTraits else None

        #Fix random seed
        self.args.randomSeed = randomSeed if randomSeed is not None else self.args.randomSeed 
        random.seed(self.args.randomSeed)
        np.random.seed(self.args.randomSeed)
        #breakpoint()
        #exampleNumber = self.store.exampleNumber if exampleNumber == -1 else exampleNumber 
        
        self.store = FileStorage(verbose=self.verbose)
        self.store.set_example(exampleNumber) #exampleNumber
        self.store.new_example(exampleNumber) #create new folder

        self.fetch = FileFetching(self.store)        
        self.store_parameters(hyperparameters_to_tweak)
        
        self.main()
        #Insert tests (assert)
        
    def get_parser(self):
        
        #https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
        
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--I', nargs='?', const=True, type=int, help = 'please give an integer number of traders', required=False, default=10000)
        #parser.add_argument('--T', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=10*self.Ty)
        parser.add_argument('--J', nargs='?', const=True, type=int, help = 'please give an integer number of stock issuers', required=False, default=1)
        #parser.add_argument('--S', nargs='?', const=True, type=int, help = 'please give an integer number of simulations', required=False, default=4)
        parser.add_argument('--randomSeed', nargs='?', const=True, type=int, help = 'please give an integer number as random seed', required=False, default=450)
        #parser.add_argument('--Ttraining', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=4*self.Ty)
        
        #parser.add_argument('--I', nargs='?', const=True, type=int, help = 'please give an integer number of traders', required=False, default=5000) #10000
        parser.add_argument('--T', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=2500) #10*self.Ty
        #parser.add_argument('--J', nargs='?', const=True, type=int, help = 'please give an integer number of stock issuers', required=False, default=1)
        parser.add_argument('--S', nargs='?', const=True, type=int, help = 'please give an integer number of simulations', required=False, default=10)
        #parser.add_argument('--randomSeed', nargs='?', const=True, type=int, help = 'please give an integer number as random seed', required=False, default=6)
        parser.add_argument('--Ttraining', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=4000 ) #self.Ty) #4*self.Ty
        
        parser.add_argument('--notes', nargs='?', const=True, type=str, help = 'insert notes', required=False)
       
        parser.add_argument('--CognitiveTraits', nargs=1, type=str, help = 'insert cognitive traits to embed: optimistic bias, relative value-shaping, impulsivity', default=['no bias individual'], required=False)  
        #parser.add_argument('--LSEstock', nargs=3, type=str, help = 'insert LSE stock abbreviation as stock price buffer to kick-off simulations for the agents', default='MONY', required=False)  
        parser.add_argument('--LSEstock', nargs=3, type=str, help = 'insert LSE stock abbreviation as stock price buffer to kick-off simulations for the agents', default='IHG', required=False) #XLOM
        
        parser.add_argument('--pretrainingFlag', dest='pretrainingFlag', action='store_true')
        parser.add_argument('--TbatchPretraining', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=6*self.Ty)
        parser.add_argument('--Spretraining', nargs='?', const=True, type=int, help = 'please give an integer number of pretraining batches', required=False, default=30)
        parser.add_argument('--tSave', nargs='?', const=True, type=int, help = 'please give an integer number time steps to save data as checkpoint', required=False, default=5000) #10*self.Ty)
        parser.add_argument('--saveMicro', dest='saveMicro', action='store_true')
        #parser.add_argument('--ex', nargs='?', const=True, type=int, help = 'please give an integer number as example to run', required=False, default=None)
        
        self.args = parser.parse_args()
        #self.args.pretrainingFlag = self.args.pretrainingFlag[0] if self.args.pretrainingFlag else self.args.pretrainingFlag
        
        print(f'\n save micro {self.args.saveMicro} \n')
  
    def setup(self):
            
        store = FileStorage(verbose=self.verbose)
        store.output_folders()
                
        fetch = FileFetching(store)
        self.Tw, self.Tm, self.Ty = fetch.time_references()
        
    def store_parameters(self, hyperparameters_to_tweak):
        
        ## SET PARAMETERS
        self.store_master()
 
        self.sample_parameters(hyperparameters_to_tweak)
        
        self.set_time_parameters() 
        self.store_master_time()
        self.store.t_init(self.tInit) #set tInit
      
#@jit(parallel=True)
    def main(self):
        
        #print(f'pretraining: {self.args.pretrainingFlag}')
        
        self.pretraining_nobias() if self.args.pretrainingFlag else None 
        #self.pretraining() if self.args.pretrainingFlag else None     
        self.training()

        ## Plot 
        #self.analysis = SimulationAnalysis(self.fetch)
        #self.analysis.plot_training()
        
        ###############
        ### TESTING ###
        ###############
        
        [self.testing(s) for s in range(self.args.S)]
                                
        #TimeStart = time.perf_counter() #Computational performance

        #pool = mp.Pool(processes=5)
        #pool.map(self.testing, range(self.args.S))
        #pool.close()
        #pool.join()

        #TimeEnd = time.perf_counter()
        #print(f'\n Computational time: {np.round(TimeEnd - TimeStart, 2)} s \n')


        ## plot  
        #self.analysis.plot_testing()
   
        ### create pdf report
        #WriteReport(simulation)
        
    def store_master(self):
        
        #See standard notation for function description
        '''
        It creates a folder for this batch of simulations, called 'Ex. n', to indicate experiment number.
        It stores the master file with I, T, J, S variables chosen in Data folder.
        It samples parameters for each trader and saves parameter data file in Data folder.
        It starts training (how long?), it resets the traders' wealth and stock prices to start S independent simulations.
        It summarises the main results in a report, showing the distribution of the stylised facts from the S simulations compared to empirical data,
        stored in Data folder.
        Inputs are: number of traders I, number time steps T, number of stocks J, number of simulations S, string-like notes. 
        '''
                
        self.store.master_file({'I': self.args.I,
                          'J':self.args.J,
                          'CognitiveTraits':self.args.CognitiveTraits,
                          'saveMicro':self.args.saveMicro,
                          'randomSeed': self.args.randomSeed,
                          'LSEstock':self.args.LSEstock,
                          'notes':self.args.notes}) ## save master file with key input variables        

    def store_master_time(self):
    
        self.store.master_file({'T':self.args.T,
                                'S':self.args.S,
                                'Ttraining':self.args.Ttraining,
                                'Straining':self.Straining,
                                'Tbatch':self.Tbatch,
                                'pretrainingFlag':self.args.pretrainingFlag,
                                'TbatchPretraining':self.args.TbatchPretraining*self.args.pretrainingFlag,
                                'Spretraining':self.args.Spretraining*self.args.pretrainingFlag,
                                'randomSeedTrain': self.args.randomSeedTrain,
                                'randomSeedTest': self.args.randomSeedTest},
                                 time=True)

    def set_time_parameters(self):
    
        # Impose maximal length for CPU memory purposes
        self.args.TbatchPretraining = min(self.args.TbatchPretraining, self.args.tSave)
        #self.args.Ttraining = min(self.args.Ttraining, tSave)
        self.args.T = min(self.args.T, self.args.tSave)
        
        remainderCondition = (np.mod(self.args.Ttraining, self.args.tSave) > 0)
        
        self.Straining = int(self.args.Ttraining / self.args.tSave)
        self.Straining += remainderCondition #add one batch if there is a remainder
        self.Tbatch = np.zeros((self.Straining)).astype(int)
        self.Tbatch[:-1] = self.args.tSave
        self.Tbatch[-1] = self.args.tSave*(1-remainderCondition)+np.mod(self.args.Ttraining, self.args.tSave)*remainderCondition
        
        tInit = self.fetch.parameters_json('SYMBA')['tInit']
        
        self.args.TbatchPretraining += tInit
        self.args.Ttraining += tInit
        self.Tbatch += tInit
        self.args.T += tInit
        
        self.tInit = tInit
        
        self.args.randomSeedTrain = list(range(self.args.randomSeed, self.args.randomSeed + self.Straining))
        self.args.randomSeedTest = list(range(self.args.randomSeedTrain[-1]+1, self.args.randomSeedTrain[-1]+1 + self.args.S))
        
    def sample_parameters(self, hyperparameters_to_tweak):
        
        SampleParameters = ParameterSampling(self.store, self.fetch, hyperparameters_to_tweak)
        SampleParameters.market()
        SampleParameters.wallet() ## save parameter file
        SampleParameters.SYMBA() ## save parameter file
                
        SampleParameters.RL()     ## save parameter file #self.args.CognitiveTraits

    def pretraining_nobias(self):
        
        for pretrainingID in range(self.args.Spretraining):
        
            self.example = pSYMBA(self.fetch)         
            self.example.pretraining_nobias(pretrainingID)
            self.store.data3(self.example, f'pretraining_{pretrainingID}')
            del self.example

    def pretraining(self):
        
        for pretrainingID in range(self.args.Spretraining):
        
            self.example = pSYMBA(self.fetch)         
            self.example.pretraining(pretrainingID)
            self.store.data3(self.example, f'pretraining_{pretrainingID}')
            del self.example
        
    def training(self):
                
        for s in range(self.Straining):

            self.example = pSYMBA(self.fetch)
                
            self.example.training(s)
            self.store.data3(self.example, f'training_{s}')
            #self.store.order_books(self.example, f'training_{s}')
        
            del self.example
        

    def testing(self, s): #def testing(self, exampleTuple):

        #s, example = exampleTuple
        
        example = pSYMBA(self.fetch)
        
        #np.random.seed(s*int(time.time())) #random seed to diversify multiprocessing
        #np.random.seed((s * int(time.time())) % 123456789)

        example.testing(s)
        self.store.data3(example, s)
        
        #del self.example
        
        

if __name__ == '__main__':
    
    Main()
    
    
    
########### Testing serially

        #for s in range(self.args.S):

        #    example = pSYMBA(self.fetch)
        #    np.random.seed((s * int(time.time())) % 123456789) #random seed to diversify multiprocessing

        #    example.testing(s)
        #    self.store.data3(example, s)    
    
    
##### No batch training
        #   self.example.training()
        #   self.store.data3(self.example)
        #   self.store.order_books(self.example)

