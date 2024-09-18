#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 12:26:36 2022

@author: svrizzi
"""

import matplotlib.pyplot as plt

import shutil
import numpy as np
import time
from tqdm import tqdm

#from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool

#Classes to run simulations
from simulate.model import Market

#########################################################################
######################       SIMULATIONS     ############################
#########################################################################

class pSYMBA:
    
    def __init__(self, fetch):
    
        self.fetch = fetch
        self.verbose = fetch.verbose
        self.tInit = fetch.parameters_json('SYMBA')['tInit']

    def welcome(self):
        
        columns = shutil.get_terminal_size().columns #to print in terminal
        
        print(f'\n ######### Initialising pSYMBA: MARL financial market simulation ({self.phase}) ######### \n'.center(columns)) if self.verbose else None

    def pretraining(self, s):

        self.phase = 'pretraining'
        self.s = s
                
        self.welcome()
        self.market = Market(self.fetch, self.phase, s)
        self.run_simulation()

    def pretraining_nobias(self, s):

        self.phase = 'pretraining_nobias'
        self.s = s
                
        self.welcome()
        self.market = Market(self.fetch, self.phase, s)
        self.run_simulation()

    def training(self, s):

        #self.s = 'training'
        self.phase = 'training'
        self.s = s
        
        self.welcome()
        self.market = Market(self.fetch, self.phase, s)
        self.run_simulation()

    def testing(self, s):

        self.phase = 'testing'
        self.s = s
                
        self.welcome() #Comment out when parallelising
        self.market = Market(self.fetch, self.phase, self.s)
        self.run_simulation()
        
    def calibrating(self):

        #self.s = 'calibrating'
        self.phase = 'calibrating'
        
        self.welcome()

        #reload variables to ensure no over-writing in threads?
        self.market = Market(self.fetch, self.phase)

        TimeStart = time.perf_counter() #Computational performance
        [self.simulation_steps(t) for t in tqdm( range(self.tInit, self.market.T) )]
        TimeEnd = time.perf_counter()
        print(f'Computational time: {np.round(TimeEnd - TimeStart, 2)} s \n') if self.verbose else None
        
        return self.market.Price

    def run_simulation(self):
        
        Trange = range(self.tInit, self.market.T)
        Trange = tqdm( Trange ) if self.market.saveMicro else Trange
        
        TimeStart = time.perf_counter() #Computational performance
        print(f'Start {self.phase} {self.s} / {self.market.S-1} \n') if self.verbose else None
        [self.simulation_steps(t) for t in Trange]
        print(f'Done with {self.phase} {self.s} / {self.market.S-1} \n') if self.verbose else None #print(f'Done with simulation {self.s} in example {self.fetch.store.flag}')
        TimeEnd = time.perf_counter()
        print(f'Computational time: {np.round(TimeEnd - TimeStart, 2)} s \n') if self.verbose else None #print(f'Done with training in example {self.fetch.store.flag}')


    ###########################################################################
    ############################# Simulation steps ############################
    ###########################################################################

    def simulation_steps(self, t):
        
        self.market.new_time_step(t) #Copy values from previous time step as the baseline for current time step
        
        self.market.decide_forecasting() ### Decision Making on Forecasting and Trading ###
        
        #Model-based RL trading
        self.market.state_trading()
        self.model_based() #if t >= self.market.tInitLearn else None     
        self.market.decide_trading()
        
        self.market.decide_pricing()
                
        [self.market.validate_trading_decisions(stock) for stock in self.market.stocks] #convert trading actions to pass when not enough available volume or margin
        
        self.market.risk_free_interests() #on available margin
        
        [ self.market.auction(stock) for stock in self.market.stocks ] ### Auction for each stock ###
        
        self.market.reference_prices_order_book() #update reference prices (best bid and ask) for next time step
        
        [self.market.cancel_orders(stock) for stock in self.market.stocks] #cancel orders to update back available margin and volume for unsold orders
        
        self.market.wallet() ### update wallets (owned stock values and available margins)
        #self.market.dividends() ### Annual stock dividends and bank interests

        self.learning_steps(t) if t >= self.market.tInitLearn else None
        
        self.market.compress_variables() if t == self.market.T-1 else None

    ### Model-based RL ###
    
    def model_based(self):
    
        [ self.market.reward_Trading(stock) for stock in self.market.stocks]
        [ self.market.relative_rescaling_trading(stock) for stock in self.market.stocks ] #if 'relative rescaling' in self.market.CognitiveTraits else None
        [ self.market.learn_trading(stock) for stock in self.market.stocks]
        
    ### Learning ###
               
    def learning_steps(self, t):
        
        [ self.market.observe_outcomes(stock) for stock in self.market.stocks ]
        
        [ self.market.relative_rescaling(stock) for stock in self.market.stocks ] #if 'relative rescaling' in self.market.CognitiveTraits else None
        
        [ self.market.learn_forecasting(stock) for stock in self.market.stocks]
        #[ self.market.learn_trading(stock) for stock in self.market.stocks]
        [ self.market.learn_pricing(stock) for stock in self.market.stocks]
        
        #[self.market.replay_trading(stock) for stock in self.market.stocks]
                
    def diagnostic_for_trading(self, t): #Diagnosis: check that cond[idx] grows (trader keeps trading) and that np.unique(...) shows the whole 3 decisions in the decision space
    
        if t > 0: #tInit+max(self.market.tau):
            cond = (abs(self.market.actionTrading[:, 0, self.tInit:t+1]-1)>0).sum(axis=1)
            idx = np.argmin(cond) #least active
            print(t, idx, cond[idx], self.market.AvailableMargin[idx, t], self.market.Price[0, t], np.unique(self.market.actionTrading[idx, 0, self.tInit:t+1])) 
            time.sleep(.2)
