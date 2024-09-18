################ Import key functions ############################

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching
from initialise.sample_parameters import ParameterSampling

from simulate.simulation import pSYMBA

from examples_analysis import ExampleAnalysis

import argparse

################ Parallel computation parameters #################

import numpy as np
import pandas as pd
import multiprocessing as mp
import time

#%%

class CreateDatasets:
    
    def __init__(self):
        
        self.Ty = 252
        
        self.verbose = False
        self.get_parser()
        
        self.set_hyperparameters_to_tweak()
        
        self.run_hyperparameter_sets()

    def set_hyperparameters_to_tweak(self):

        # !!! Remember to deactive bias in args.pars
        #self.N = 4
        #self.hyperparameterSets = [{'Liquidity balance': np.linspace(0.5, 2.5, self.N)}]
        #self.hyperparameterSets = [{'Qs': np.logspace(0, self.N-1, self.N)}]
        #self.hyperparameterSets = [{'gShape': np.linspace(1, 13, self.N)}]
        #self.hyperparameterSets = [{'tauShape': np.linspace(1, 13, self.N)}]
        #self.hyperparameterSets = [{'bFlat': [0, 1, 4, 10]}]
        #self.hyperparameterSets = [{'bFlat': [0, 1, 4, 10],
        #                            'Liquidity balance': np.ones(self.N)*LiquidityBalance}
        #                            for LiquidityBalance in [.5, 1, 2, 5]]
        #self.hyperparameterSets = [{'arhoAvailableMargin': [1, 2.5, 5, 10],
        #                            'Liquidity balance': np.ones(self.N)*LiquidityBalance}
        #                            for LiquidityBalance in [.5, 1, 2, 5]]


        self.N = 5
        
        #!!! Remember to active bias in args.pars        
        #self.hyperparameterSets = [{'alphaDisc': np.linspace(0, 1, self.N),
        #                            'alphaConf': np.ones(self.N)*alpha,
        #                            'distribution': [False]*self.N,
        #                            'kernel':[False]*self.N}
        #                            for alpha in np.linspace(0, 1, self.N)]
        
        
        if self.args.ConfirmationBiasAnalysis:
            
            columns = ['aAlphaConf', 'bAlphaConf', 'aAlphaDisc', 'bAlphaDisc'] if self.distribution else ['alphaDisc', 'alphaConf']
            
            from itertools import product
        
            if self.distribution:
                        
                abAlphas = [2, 3, 4.5, 10]
            
                dfMain = pd.DataFrame(list(product(abAlphas, abAlphas, abAlphas, abAlphas)), columns=columns)
                
                # No bias, uniform distribution across traders
                df0 = pd.DataFrame(data=[[1, 1, 1, 1]], columns=columns)
                
                # No bias, symmetric bell-distribution across traders
                #df1 = pd.DataFrame(data=[[2, 2, 2, 2]], columns=columns)
            
                # Optimistic bias,
                
                #data = [[3, 2, 2, 2], #bias (+), right-skwed distribution vs symmetric
                #        [3, 2, 3, 2], #no bias, right-skwed distributions
                #        [2, 3, 2, 3], #no bias, left-skwed distributions
                #        [2, 3, 2, 2], #bias (-), left-skwed distribution vs symmetric
                #        [2, 2, 2, 3], #bias (+), symmetric vs left-skwed distribution
                #        [2, 2, 3, 2], #bias (-), symmetric vs right-skwed distribution
                        
                #        [4.5, 2, 2, 2], #bias (+), right-skwed distribution vs symmetric
                #        [4.5, 2, 4.5, 2], #no bias, right-skwed distributions
                #        [2, 4.5, 2, 4.5], #no bias, left-skwed distributions
                #        [2, 4.5, 2, 2], #bias (-), left-skwed distribution vs symmetric
                #        [2, 2, 2, 4.5], #bias (+), symmetric vs left-skwed distribution
                #        [2, 2, 4.5, 2], #bias (-), symmetric vs right-skwed distribution
                
                df = pd.concat([df0, dfMain], ignore_index=True)
            
                df['distribution'] = self.distribution
                df['kernel'] = False
            
            elif not self.distribution:
                  
                ### Low alphas ###
                #alphas = [0.05, 0.25, 0.5, 0.75, 1] #np.linspace(0, 1, self.N)
                #alphas = [0.001, 0.005, 0.01, 0.1] #np.linspace(0, 1, self.N)
                #alphas = [0.01, 0.02, 0.03, 0.04, 0.05] #np.linspace(0, 1, self.N)
                #alphas = [0.001, 0.002, 0.003, 0.004, 0.005] #np.linspace(0, 1, self.N)
                #alphas = [0.1, 0.2, 0.3, 0.4, 0.5] #np.linspace(0, 1, self.N)
                #alphas = [0.001]
                alphas = [0.0, 0.25]
                
                #alphas = [0.01, 0.02]
                alphaDisc = alphas
                alphaConf = alphas
                dfLow = pd.DataFrame(list(product(alphaDisc, alphaConf)), columns=columns)
            
                ### High alphas ###
                #alphas = [0.25, 0.5, 0.75, 1]
                #alphas = [0.06, 0.07, 0.08, 0.09, 0.1]
                #alphas = [0.007, 0.0085, 0.01]
                #alphas = [0.01]
                #alphas = [0.5, 0.75, 0.1] #np.linspace(0, 1, self.N)
                #alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1] #np.linspace(0, 1, self.N)
                #alphas = [0.06, 0.07]
                alphas = [0.5, 0.75, 1]
                alphaDisc = alphas
                alphaConf = alphas
                dfHigh = pd.DataFrame(list(product(alphaDisc, alphaConf)), columns=columns)
            
                # Noise-traders condition (no learning)
                dfNoise = pd.DataFrame(data=[[0, 0]], columns=columns)
            
                df = pd.concat([dfNoise, dfLow, dfHigh], ignore_index=True)
                #df = pd.concat([dfNoise, dfLow], ignore_index=True)
                #df = pd.concat([dfNoise, dfHigh], ignore_index=True)
            
                df['distribution'] = self.distribution
                df['kernel'] = False
            
                repetitions = 5
                df = df.loc[df.index.repeat(repetitions)].reset_index(drop=True)
                
        if not self.args.ConfirmationBiasAnalysis:
            
            #self.hyperparameterSets = [{'bFlat': [0, 1, 4, 10]}]
            #df = pd.DataFrame(data=[0, 1, 4, 10, 50], columns=['bFlat'])
            df = pd.DataFrame(data=[1, 2, 5, 10], columns=['arhoAvailableMargin'])  
            repetitions = 10 #5
            df = df.loc[df.index.repeat(repetitions)].reset_index(drop=True)
            
        self.hyperparameterSets = [df.to_dict('list')]
        self.N = len(df.index)
            
        self.df = df
            
        #breakpoint()        

        #self.hyperparameterSets = [{'alphaConf': np.linspace(0, 1, self.N),
        #                            'distribution':[False]*self.N,
        #                            'kernel':[False]*self.N}]
        
        #self.hyperparameterSets = [{'aAlphaConf': [0.5, 1.1, 2.6, 5, 10],
        #                            'distribution': [True]*self.N,
        #                            'kernel':[False]*self.N}
        #                            for alpha in np.linspace(0, 1, self.N)]
        
        #self.N = 6
        #self.hyperparameterSets = [{'tauScale': [1, 2, 4, 8, 12, 16]}]

        #self.hyperparameters = [{'aAlphaConf': np.random.rand(self.N)*20,
        #                        'bAlphaConf': np.random.rand(self.N)*20,
        #                        'aAlphaDisc': np.random.rand(self.N)*20,
        #                        'bAlphaDisc': np.random.rand(self.N)*20}]

    def run_hyperparameter_sets(self):

        print('\n Run in parallel')        

        if len(self.hyperparameterSets) == 1: #parallel examples (1 example = 1 hyperparameter set)
            
            hyperparameters = self.hyperparameterSets[0]
            hyperparameters = self.prepare_folders_and_hyperparameters(hyperparameters)
            
            pool = mp.Pool(processes=12)
            pool.imap(self.main2, enumerate(hyperparameters))
            pool.close()
            pool.join()

        else: # parallelise datasets (groups of examples)

            pool = mp.Pool(processes=12)
            pool.imap(self.main3, enumerate(self.hyperparameterSets))
            pool.close()
            pool.join()

        print('\n Done')

    def prepare_folders_and_hyperparameters(self, hyperparameters):

        dataFolderNameStrings = []
        
        for key, value in hyperparameters.items():
        
            value = np.unique(value)
            
            if len(value) > 1:
                dataFolderNameStrings.append(key+f'-{value[0]}-{value[-1]}')
            else: 
                dataFolderNameStrings.append(key+f'-{value[0]}')
                
        datasetFolderName = ' '.join(dataFolderNameStrings)
        datasetFolderName = datasetFolderName+' '+self.args.notes if self.args.notes!='' else datasetFolderName
        datasetFolderName = 'ConfirmationBiasAnalysis '+datasetFolderName if self.args.ConfirmationBiasAnalysis else datasetFolderName
                 
        self.store = FileStorage(datasetFolder=datasetFolderName, verbose=self.verbose)
        self.store.output_folders()
        self.store.parameters(hyperparameters, 'hyperparameters') #save hyperparameters in json
        
        [self.store.new_example() for _ in range(self.N)]
        
        hyperparametersT = [dict(zip(hyperparameters, keys)) for keys in zip(*hyperparameters.values())] #transpose dictionary
        
        #Save csv for reference in case of confirmation bias analysis
        self.df.to_csv(self.store.DataFolderPath+'/hyperparameters.csv') if self.args.ConfirmationBiasAnalysis else None
        
        return hyperparametersT

    def main3(self, hyperparametersSetTuple):
    
        self.index, hyperparameters = hyperparametersSetTuple
        hyperparameters = self.prepare_folders_and_hyperparameters(hyperparameters)

        [self.main2(hyperparametersTuple) for hyperparametersTuple in enumerate(hyperparameters)]

    def main2(self, hyperparametersTuple):

        self.index, hyperparameters = hyperparametersTuple
          
        ## SET PARAMETERS
        self.store_and_fetch_example()
        
        SampleParameters = ParameterSampling(self.store, self.fetch)
        
        SampleParameters.market(hyperparameters_to_tweak=hyperparameters)
        SampleParameters.wallet(hyperparameters_to_tweak=hyperparameters) ## save parameter file
        SampleParameters.SYMBA(hyperparameters_to_tweak=hyperparameters) ## save parameter file
        SampleParameters.RL(self.args.CognitiveTraits, hyperparameters_to_tweak=hyperparameters) ## save parameter file
        
        np.random.seed((self.index * int(time.time())) % 123456789)
        
        self.training()
        self.testing()
        
    def get_parser(self):
        
        #https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
        
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--I', nargs='?', const=True, type=int, help = 'please give an integer number of traders', required=False, default=100)
        parser.add_argument('--T', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=10*self.Ty)
        parser.add_argument('--J', nargs='?', const=True, type=int, help = 'please give an integer number of stock issuers', required=False, default=1)
        parser.add_argument('--S', nargs='?', const=True, type=int, help = 'please give an integer number of simulations', required=False, default=20)
        parser.add_argument('--Ttraining', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=4*self.Ty)
        parser.add_argument('--notes', nargs='?', const=True, type=str, help = 'insert notes', required=False)
       
        #parser.add_argument('--CognitiveTraits', nargs=1, type=str, help = 'insert cognitive traits to embed: optimistic bias, relative value-shaping, impulsivity', default=['optimistic bias', 'relative rescaling', 'counterfactual'], required=False) 
        #parser.add_argument('--CognitiveTraits', nargs=1, type=str, help = 'insert cognitive traits to embed: optimistic bias, relative value-shaping, impulsivity', default=['no bias', 'relative rescaling', 'counterfactual'], required=False)
        parser.add_argument('--CognitiveTraits', nargs=1, type=str, help = 'insert cognitive traits to embed: optimistic bias, relative value-shaping, impulsivity', default=['no bias', 'counterfactual'], required=False)
        #parser.add_argument('--CognitiveTraits', nargs=1, type=str, help = 'insert cognitive traits to embed: optimistic bias, relative value-shaping, impulsivity', default=['optimistic bias', 'counterfactual'], required=False)  
        #parser.add_argument('--LSEstock', nargs=3, type=str, help = 'insert LSE stock abbreviation as stock price buffer to kick-off simulations for the agents', default='MONY', required=False)  
        parser.add_argument('--LSEstock', nargs=3, type=str, help = 'insert LSE stock abbreviation as stock price buffer to kick-off simulations for the agents', default='SXS', required=False) #XLOM
        
        parser.add_argument('--pretrainingFlag', nargs=1, type=bool, help = 'pre-training from LSE imposed price', default=False, required=False) #XLOM
        parser.add_argument('--TbatchPretraining', nargs='?', const=True, type=int, help = 'please give an integer number of time steps', required=False, default=6*self.Ty)
        parser.add_argument('--Spretraining', nargs='?', const=True, type=int, help = 'please give an integer number of pretraining batches', required=False, default=30)
        parser.add_argument('--tSave', nargs='?', const=True, type=int, help = 'please give an integer number time steps to save data as checkpoint', required=False, default=100*self.Ty)
        
        parser.add_argument('--ConfirmationBiasAnalysis', nargs='?', type=bool, help = '', default=False, required=False) 
              
        self.args = parser.parse_args()
       
        if self.args.ConfirmationBiasAnalysis:
        
            self.args.CognitiveTraits = ['optimistic bias']
            [print(f'{cognitiveTrait} set to ON to analyse its effect') for cognitiveTrait in self.args.CognitiveTraits]
            self.distribution = False
        
        self.args.Ttraining += tInit
        self.args.T += tInit
        
    def store_and_fetch_example(self):
        
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

        self.store.set_example(self.index)
        
        self.store.master_file(I=self.args.I,
                          T=self.args.T,
                          J=self.args.J, 
                          S=self.args.S,
                          Ttraining=self.args.Ttraining,
                          CognitiveTraits=self.args.CognitiveTraits,
                          LSEstock=self.args.LSEstock,
                          notes=self.args.notes) ## save master file with key input variables        

        self.fetch = FileFetching(self.store)
        
    def training(self):
        
        example = pSYMBA(self.fetch)
        example.training()
        self.store.data3(example)
        self.store.order_books(example)

    def testing(self):
        
        for s in range(self.args.S):
            
            example = pSYMBA(self.fetch)
            example.testing(s)
            self.store.data3(example, s)


if __name__ == '__main__':
    
    CreateDatasets()
    #ExampleAnalysis(datasets= 'all')
