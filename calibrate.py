#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:27:12 2022

@author: svrizzi
"""

import numpy as np
from initialise.time_parameters import tInit, Tw, Ty
from scipy.stats import skew, kurtosis
from initialise.sample_parameters import ParameterSampling
from simulate.simulation import pSYMBA

import argparse
from scipy import optimize

class Calibrate:

    def __init__(self):
                
        self.get_parser()

        self.load_data()
        self.GT_Returns = self.returns(self.GT_Price)
        
        self.n_SF = 4
        
        self.GT_SF = self.stylised_facts(self.GT_Returns)
        
        self.B = 5000 #bootstrap parameter
        self.W = np.linalg.inv(self.Sigma_hat())
        
        self.calibrate()
        
        
    def get_parser(self):
        
        #https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
        
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--task', nargs='?', const=True, type=str, help = 'please select the task LSE or recovery', required=False, default='recovery')
        parser.add_argument('--exampleNumber', nargs='?', const=True, type=int, help = 'please give an integer number identifying an example already run', required=False, default=-1)
        
        self.args = parser.parse_args()
        
    def load_data(self):
        
        if self.args.task=='recovery':
            
            from filemanagement.store import FileStorage
            from filemanagement.fetch import FileFetching

            self.Store = FileStorage()
            self.Store.set_example(self.args.exampleNumber)
            self.Fetch = FileFetching(self.Store)
            
            self.CognitiveTraits = self.Fetch.input_parameters()['CognitiveTraits']
            
            self.GT_Price = self.Fetch.saved('P')[0, tInit:]
        
    
    def calibrate(self):
            
        NumberOfHyperparameters = 4
        TotalGuesses = 5
        
        InitialGuess = np.zeros((TotalGuesses, NumberOfHyperparameters))
        InitialGuess = np.random.rand(TotalGuesses, NumberOfHyperparameters)*5
        bnds = ((0.1, None), (0.1, None), (0.1, None), (0.1, None))
        results = np.zeros(TotalGuesses)
        distances = np.zeros(TotalGuesses)
        
        for GuessNumber in range(TotalGuesses):        

            result = optimize.minimize(self.distance, InitialGuess[GuessNumber], args='g', bounds=bnds, method='L-BFGS-B') #constraints=con, method='SLSQP')
            print(f'Optimised hyperparameters run {GuessNumber}:', result.x)
        
            results[GuessNumber]  = result.x
            print('Success: ', result.success)
            distances[GuessNumber] = result.fun
        
        OptimisedHyperparameters = results[np.argmin(distances)]
        
        print('Optimised hyperparameters:', OptimisedHyperparameters)

    def returns(self, Price):
        
        return Price[1:] - Price[:-1]
        
    def stylised_facts(self, R):
        
        sf = np.zeros(self.n_SF)
        
        #breakpoint()
        
        sf[0] = R.mean()
        sf[1] = R.var()
        sf[2] = skew(R)
        sf[3] = kurtosis(R)
        #sf[4] = sm.tsa.acf(R, nlags=1)[1]
        
        return sf

    def bootstrap(self):
    
        Tchuncker = Tw    
    
        n_blocks = int(len(self.GT_Returns)/Tchuncker)
        T_bootstrap = n_blocks*Tchuncker
        
        data_blocks =  np.reshape(self.GT_Returns[:T_bootstrap], [n_blocks, Tchuncker])
        Returns_bootstrap = np.zeros((self.B, Tchuncker*n_blocks))
        bootstrap_SF = np.zeros((self.B, self.n_SF))
        
        for bootstrap in range(self.B):
            
            for block in range(n_blocks):
                draw_block = np.random.choice(range(n_blocks))
                Returns_bootstrap[bootstrap, Tchuncker*block: Tchuncker*(block+1)] = data_blocks[draw_block]
            
            #bootstrap_Returns = self.returns0(P_bootstrap[bootstrap])
            bootstrap_SF[bootstrap] = self.stylised_facts(Returns_bootstrap[bootstrap])
                        
        return bootstrap_SF
    
    def Sigma_hat(self):
        
        bootstrap_SF = self.bootstrap()
        bootstrap_SF -= bootstrap_SF.mean(axis=0) #normalise
    
        return 1/self.B * (bootstrap_SF.T @ bootstrap_SF)

        
    def distance(self, hyperparameters, *parameterNames):
        """objective function, to be solved."""
        
        gShape, gScale = hyperparameters[0], hyperparameters[1]
        
        print(f'Trying: {gShape, gScale}')
        
        SampleParameters = ParameterSampling(self.Store, self.Fetch, calibrating=True)
        SampleParameters.SYMBA(hyperparameters=hyperparameters)    ## save parameter file
        
        SampleParameters.RL(self.CognitiveTraits)     ## save parameter file
        
        example = pSYMBA(self.Fetch) #could use the testing to run mutiple times and takes statistics   
        
        P = example.calibrating()
        Returns = self.returns(P[0])
        SF = self.stylised_facts(Returns)
        
        L = (self.GT_SF - SF).T @ self.W @ (self.GT_SF - SF)
        return L


if __name__ == '__main__':
    
    Calibrate()