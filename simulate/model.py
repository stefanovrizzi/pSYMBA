#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:55:13 2022

@author: svrizzi
"""

# Find faster alternatives to np.where

import time

import numpy as np
import pandas as pd
import random
#from numba import njit

import matplotlib.pyplot as plt

from scipy.special import expit, softmax
from sklearn.linear_model import LinearRegression
#from scipy.stats import linregress
from scipy import stats

import datetime

from statsmodels.tsa.stattools import acf
import statsmodels.api as sm

import sys
ExpOverFLowLimitMax = np.log(sys.float_info.max / 2) #to avoid overflow in exponential SoftMax Decision Rule
ExpUnderFLowLimitMax = np.log(sys.float_info.min * 2) #to avoid underflow in exponential SoftMax Decision Rule
OverFLowLimitMin = sys.float_info.min #to avoid overflow in exponential SoftMax Decision Rule

#%% For diagnosticsf

np.seterr(all='raise')

#%%

class Market:
    
    """
    A class to run the financial market.
    """
    
    def __init__(self, fetch, phase, s=None):
        
        """
        Constructs all the necessary attributes for the time step and stock object.
        """

        self.phase = phase
        self.fetch = fetch
        
        #Input parameters
        inputParameters = fetch.input_parameters()

        I = inputParameters['I']
        J = inputParameters['J']
        
        timeParameters = fetch.input_parameters(time=True)
        Tbatch = timeParameters['Tbatch'][s] if phase=='training' else None #training length
        T = timeParameters['TbatchPretraining'] if phase in ['pretraining', 'pretraining_nobias'] else Tbatch if (phase=='training' or phase=='calibrating') else timeParameters['T'] if phase=='testing' else None
        
        self.S = timeParameters['Spretraining'] if phase in ['pretraining', 'pretraining_nobias'] else timeParameters['Straining'] if phase=='training' else timeParameters['S'] if phase=='testing' else None
                
        pretrainingFlag = timeParameters['pretrainingFlag']

        randomSeed = timeParameters['randomSeedTrain'][s] if phase in ['pretraining', 'pretraining_nobias', 'training'] else timeParameters['randomSeedTest'][s]
        random.seed(randomSeed)
        np.random.seed(randomSeed)

        fetch.set_folder_name(phase, s) #folder to recall previous variables
                        
        self.I = I
        self.J = J
        self.T = T

        self.saveMicro = inputParameters['saveMicro']
        Tmicro = self.T if self.saveMicro else 1
        TBinary = self.T if self.saveMicro else 2
        
        self.Tmicro = Tmicro

        self.Tw, self.Tm, self.Ty = fetch.time_references()
        
        self.verbose = False #True
        
        warning_subfolder = s if phase == 'testing' else phase+f'_{s}' if phase in ['training', 'pretraining'] else breakpoint()
        self.log_file = fetch.simulation_folder_path(warning_subfolder)+'/warnings_log.txt'
        
        ################################################
        #################### MARKET ####################
        ################################################
        
        marketParameters = fetch.parameters_json('market')
        
        self.tick = marketParameters['tick']        
        self.b = marketParameters['b']
        self.bFlat = marketParameters['bFlat']
        self.D = marketParameters['D']
        self.RiskFreeRate = marketParameters['RiskFreeRate']
        self.RiskFreeRate = (1+self.RiskFreeRate)**(1/self.Ty) -1
                
        self.stocks = list(range(J))
        self.traders = list(range(I))
        self.tradersArray = np.array(self.traders)

        self.Price = np.zeros((J, T)) #price history
        self.Volume = np.zeros((J, T)).astype(int) #trading volume
        self.BidAskSpread = np.zeros((J, T)) #spread between best bid and ask
        
        self.returns = np.zeros((J, T-1))

        self.fundamentalValueTrue = np.zeros((J, T))
        self.fundamentalValue = np.zeros((I, J, T))

        ### Order Books for auction

        self.set_order_books() #Start market with outstanding stocks on OrderBookAsks?

        self.safeMarginPrices = 10*self.tick
        
        self.priceDictionary = np.zeros((I, J, 2))

        ###################################################
        #################### COGNITION ####################
        ###################################################
        
        self.CognitiveTraits = inputParameters['CognitiveTraits']
        
        #####################
        ### RL parameters ###
        #####################
        
        RLParameters = fetch.parameters_csv('RL')
        
        self.beta = RLParameters['beta'].values #if phase=='training' or phase=='calibrating' else RLParameters['beta']*10 if phase=='testing' else None
        self.alphaConf = RLParameters['alphaConf'].values
        self.alphaDisc = RLParameters['alphaDisc'].values
        self.alphaV = RLParameters['alphaV'].values

        ########################
        ### SYMBA parameters ###
        ########################
        
        SYMBAParameters = fetch.parameters_json('SYMBA')

        self.mu = SYMBAParameters['mu'] #annual growth rate
        self.sigma = SYMBAParameters['sigma'] #annual volatility rate

        #Find root for nu0
        #from scipy.optimize import fsolve
        #f = lambda x: expit(x)*(1-expit(x)) / (expit(0)*(1-expit(0))) #*2 -1
        #rewardForecastRoot = fsolve(f, [1.765])
        
        nuRoot = 1.76274717
        
        nu_plus_d = SYMBAParameters['nu_plus_d']
        nu_minus_d = SYMBAParameters['nu_minus_d']
        nu_zero_d = SYMBAParameters['nu_zero_d']
        self.nu_plus = nuRoot / nu_plus_d
        self.nu_minus = nuRoot / nu_minus_d 
        self.nu_zero = nuRoot / nu_zero_d        
        #self.expitNorm = (expit(0)*(1-expit(0)))
        
        self.sigmoid_prime_plus_norm = self.sigmoid_prime(0, self.nu_plus)
        self.sigmoid_prime_minus_norm = self.sigmoid_prime(0, self.nu_minus)
        self.sigmoid_prime_zero_norm = self.sigmoid_prime(0, self.nu_zero)
        
        self.etaHalf = SYMBAParameters['etaHalf']
        self.rewardMultiplier = SYMBAParameters['rewardMultiplier']
        self.tInit = SYMBAParameters['tInit']
        self.tInitmicro = self.tInit if self.saveMicro else 1
        self.tInitBinary = self.tInit if self.saveMicro else 2

        self.t = self.tInit-1
        
        SYMBAParameters = fetch.parameters_csv('SYMBA')
        self.tau = SYMBAParameters['tau'].values        
        self.tauUnique = set(self.tau)
        self.maxTau = max(self.tau)
                
        self.pvaluesSlope = np.zeros(self.I)

        TForecast = self.T if self.saveMicro else max(self.tau)+1
        self.tInitForecast = self.tInit if self.saveMicro else max(self.tau)+1
        
        self.rhoAsk = SYMBAParameters['rhoAsk'].values
        self.rhoBid = SYMBAParameters['rhoBid'].values
        self.g = SYMBAParameters['g'].values
        self.gSkewness = SYMBAParameters['gSkewness'].values
        #self.DrawdownLimit = SYMBAParameters['DrawdownLimit']        
        self.kappa = SYMBAParameters['kappa'].values

        self.tInitLearn = self.tInit
        
        if self.phase in ['pretraining_nobias', 'pretraining', 'calibrating', 'training']:
        
            TbatchesEffective = timeParameters['Tbatch'][:s]
            totalTbatchesEffective = sum( [TbatchesEffective[s_]-self.tInit for s_ in range(s)] )
            
        elif self.phase in ['testing']:
        
            totalTbatchesEffective = timeParameters['Ttraining']-self.tInit
        
        self.tInitLearn += ( max(self.tau)-totalTbatchesEffective )

        #####################
        ### RL variables ####
        #####################
            
        #Forecasting
        stateDimensionalityForecasting = 6 #3
        self.NactionsForecasting = 2 #3
        self.stateForecasting = np.zeros((I, J, TForecast)).astype(int)
        self.actionForecasting = np.zeros((I, J, T)).astype(int)
        self.QForecasting = np.zeros((I, Tmicro, stateDimensionalityForecasting, self.NactionsForecasting))
        self.VForecasting = np.zeros((I, Tmicro, stateDimensionalityForecasting))      
        self.RewardForecasting = np.zeros((I, J, Tmicro, self.NactionsForecasting))#.astype(int)
        
        self.knowledgeWaste = np.zeros((I, J, Tmicro))
                    
        #Trading
        self.stateTrading = np.zeros((I, J, Tmicro, 2)).astype(int)
        self.actionTrading = np.zeros((I, J, Tmicro)).astype(int) # trading action
        self.actionTradingChosen = np.zeros((I, J, Tmicro)).astype(int) # trading action
        #self.actionTradingEffective = np.ones((I, J, T)).astype(int) # actual trading action (may loose the auction)  
        self.NactionsTrading = 3    
        
        NforecastingDummy = 1 #self.NactionsForecasting
          
        self.QTrading = np.zeros((I, Tmicro, NforecastingDummy, 8, self.NactionsTrading)) #bought stocks or PL level, forecast level, volatility level, 3 actions (sell, hold, buy)
        self.VTrading = np.zeros((I, Tmicro, NforecastingDummy, 8)) #bought stocks or PL level, forecast level, volatility level
        self.RewardTrading = np.zeros((I, J, Tmicro, self.NactionsTrading)) #.astype(int)
        
        self.NtradingStates = np.size(self.QTrading[0, 0, 0, :, 0])
                
        #Order pricing
        self.NactionsPricing = 5 #5 #3
        self.NactionsPricingLimit = 4
        self.statePricing = np.zeros((I, J, Tmicro, 2)).astype(int)
        self.actionPricing = np.zeros((I, J, Tmicro)).astype(int)
        self.QPricing = np.zeros((I, Tmicro, 2, 2, self.NactionsPricing)) #bid or ask, volatility level, chosen reference price (excluding market price, it cannot be priced) -> 4 price references x 3 future order price levels deflections + market order ##+3*(self.NactionsForecasting-1)
        self.VPricing = np.zeros((I, Tmicro, 2, 2)) #bid or ask, volatility level, chosen reference price (excluding market price, it cannot be priced) -> 4 future order price levels deflections

        self.RewardPricing = np.zeros((I, J, Tmicro, self.NactionsPricing))

        self.FailedLimitOrderReward = -1
        
        self.eta = 1 / self.etaHalf #parameter for exponential decay in pricing return for limit orders (/ % of miss to get half reward)
        self.etaResolved = self.eta*np.log(2) #np.log(2) #np.log(3)
        self.etaFail = self.eta*np.log(2) #parameter for exponential decay in pricing return for limit orders (/ % of miss to get half reward)
                
        #self.eta /= self.g #np.ones(self.I) #self.g
        ###################################################
        #################### Forecast #####################
        ###################################################

        self.Forecast = np.zeros((I, J, TForecast, self.NactionsForecasting))
        
        self.TimeHorizonMask = np.zeros((len(self.traders), max(self.tau[self.traders]))).astype(bool)
        self.TimeHorizonMask[ self.traders, -self.tau[self.traders] ] = True

        ###################################################
        #################### WALLET #######################
        ###################################################
        
        #####################
        # Wallet parameters #
        #####################
        
        walletParameters = fetch.parameters_json('wallet')
        self.InitialVolumeOwned = walletParameters['InitialVolumeOwned']
        LiquidityBalance = walletParameters['Liquidity balance']
	
        ####################
        # Wallet variables #
        ####################
	
        self.VolumeOwned = np.zeros((I, J, TBinary)).astype(int) #quantity of stocks
        self.AvailableVolume = np.zeros((I, J, TBinary)).astype(int) #quantity of stocks available to sell        
        self.Aequity = np.zeros((I, J, TBinary))        

        self.Wallet = np.zeros((I, TBinary))        
        self.AvailableMargin = np.zeros((I, TBinary))
                
        self.averageBuyPrice = np.zeros((I, J, TBinary)) #weighted average price at which the trader bought the stocks since last closure

        self.resolvedTrading = np.zeros((I, J, Tmicro)) #binary
        self.resolvedAsk = np.zeros((I, J, Tmicro)) #binary
        self.resolvedBid = np.zeros((I, J, Tmicro)) #binary
        
        self.TradingDecisionMakersBinaryMaster = np.zeros((I, T)).astype(bool) #binary: active choice or not
        TradingDecisionMakersProbabilities = np.random.rand( len(self.traders),  T)
        self.TradingDecisionMakersBinaryMaster = ( TradingDecisionMakersProbabilities.T < np.ones_like(TradingDecisionMakersProbabilities).T * 1/self.tau[self.traders] ).T
        self.TradingDecisionMakersBinaryMaster[:, :self.tInit] = False
                
        self.BalanceSheet = np.zeros((I, J, Tmicro, 2)) # trading expenses, trading gains and interest rate
        self.BalanceSheetBank = np.zeros((I, Tmicro)) # interest rate
        
        self.BalanceSheetFees = np.zeros((I, J, Tmicro))
        
        ###################################################
        ######### ASSIGN VALUES TO VARIABLES ##############
        ###################################################
        
        if phase in ['pretraining', 'pretraining_nobias', 'calibrating'] or (phase == 'training' and s == 0):
            
            #####################
            ###### Market #######
            #####################
            
            self.LSE_reference_stock(fetch)
            
            self.Price[:, :self.tInit] = self.PriceLSEInit.T

            self.returns[:, :self.tInit-1] = np.log(self.Price[:, 1:self.tInit])-np.log(self.Price[:, :self.tInit-1]) #stock price returns
            self.BidAskSpread[:, :self.tInit] = self.tick #spread between overall bids and asks # initialise BidAskSpread

            #For fundamental values
            baseline = self.PriceLSE.iloc[0]

            self.fundamentalValueTrue = self.create_stochastic_process(t0=0, baseline=baseline)

            nu = np.zeros((I, T-1)) #np.random.normal(mu, sigma, size=[I]) # co-integration noise                        
            #sigma = (max(self.kappa) / self.kappa).reshape(-1, 1) * self.fundamentalValueTrue[:, 1:] / 10000
            #nu = np.random.normal(0, sigma, size=[I, T-1]) # co-integration noise
                    
            for stock in self.stocks:
                
                #epsilon_zero = np.random.normal(0, scale=baseline/3*np.exp(-self.kappa), size=self.I)
                #epsilon_zero = (baseline*self.kappa-baseline)*np.sign(np.random.rand(self.I)-.5)
                epsilon_zero = baseline*np.sign(np.random.rand(self.I)-.5)

                self.fundamentalValue[:, stock, 0] = baseline + epsilon_zero
                self.fundamentalValue[:, stock, 0] = np.maximum(self.fundamentalValue[:, stock, 0], self.tick)
                #breakpoint()
                for t in range(1, T):
                    self.fundamentalValue[:, stock, t] = self.fundamentalValue[:, stock, t-1] + nu[:, t-1] + self.kappa * (self.fundamentalValueTrue[stock, t]*np.ones(I)-self.fundamentalValue[:, stock, t-1])
                    self.fundamentalValue[:, stock, t] = np.maximum(self.fundamentalValue[:, stock, t], self.tick)

            #[ plt.plot(self.fundamentalValue[i, 0, :], alpha=0.5, linewidth=-np.log(self.kappa[i])/max(-np.log(self.kappa))) for i in range(self.I) ], plt.plot(self.fundamentalValueTrue[0], color='k', linewidth=2), plt.show()
            #breakpoint()           
            self.bestAskPrice[:] = self.Price[:, self.t]+self.tick #they won't get updated before first auction, so consider t+1, as now is tInit-1
            self.bestBidPrice[:] = self.Price[:, self.t]-self.tick
            self.reference_prices_order_book()

            #####################
            ### RL variables ####
            #####################            
            
            if (phase == 'training' and pretrainingFlag) or (phase == 'pretraining' and s > 0):
            
                self.load_Q_values()
                #self.volatilities = self.fetch.saved('volatilities').tolist()
                #self.volatilityReference[:] = self.fetch.saved('volatilityReference')
                
            #elif (phase == 'training' and not pretrainingFlag) or (phase in ['pretraining', 'pretraining_nobias'] and s == 0):
            
            #    self.initialise_volatility()
            #    self.volatilityReference[:] = [np.median(self.volatilities[stock]) for stock in self.stocks]

            ####################
            # Wallet variables #
            ####################
            
            self.VolumeOwned[:, :, :self.tInitBinary] = self.InitialVolumeOwned
            self.AvailableVolume[:, :, :self.tInitBinary] = self.VolumeOwned[:, :, :self.tInitBinary]
            self.Aequity[:, :, :self.tInitBinary] = self.VolumeOwned[:, :, :self.tInitBinary]*np.broadcast_to( self.PriceLSEInit.T[self.tInit-self.tInitBinary:self.tInit], (I, J, self.tInitBinary) )
            #self.Aequity[:, :, :self.tInit] = self.VolumeOwned[:, :, :self.tInit]*self.Price[:, :self.tInit]
            self.Wallet[:, :self.tInitBinary] = np.sum(self.Aequity[:, :, :self.tInitBinary], axis=1)*LiquidityBalance #abs( np.random.normal(0, scale=10**4, size=I) #risk-free asset value
            self.AvailableMargin[:, :self.tInitBinary] = np.copy( self.Wallet[:, :self.tInitBinary] )
            
            self.averageBuyPrice[:, :, :self.tInitBinary] = self.Price[:, self.tInit-1]
                        
        elif phase == 'testing' or (phase == 'training' and s > 0):

            #reset time from timescale in previous data file
            modtime = -timeParameters['Tbatch'][s-1] if phase == 'training' else -timeParameters['Tbatch'][-1]
            modtime += self.tInit

            #####################
            ###### Market #######
            #####################
            
            #self.LSE_reference_stock(fetch) #for reference margin only
            
            self.Price[:, :self.tInit] = fetch.saved('Price')[:, -self.tInit:] #Price
            self.Volume[:, :self.tInit] = fetch.saved('Volume')[:, -self.tInit:] #volume
            self.BidAskSpread[:, :self.tInit] = fetch.saved('BidAskSpread')[:, -self.tInit:] #spread
            self.returns[:, :self.tInit-1] = np.log(self.Price[:, 1:self.tInit])-np.log(self.Price[:, :self.tInit-1]) #stock price returns
                        
            self.fundamentalValueTrue[:, :self.tInit] = fetch.saved('fundamentalValueTrue')[:, -self.tInit:] # true fundamental value            
            t0 = self.tInit-1
            baseline = self.fundamentalValueTrue[:, t0]
            
            self.fundamentalValueTrue[:, t0:] = self.create_stochastic_process(t0=t0, baseline=baseline)
            
            #if self.saveMicro:
            self.fundamentalValue[:, :, :self.tInit] = fetch.saved('fundamentalValue')[:, :, -self.tInit:]  # fundamental value
            #else:
            #    self.fundamentalValue[:, :, self.tInit-1] = fetch.saved('fundamentalValue')[:, :, -1]  # fundamental value (compressed size), only need last value to cointegrate, but need a whole year to computed median gap in forecasting state!
            
            nu = np.zeros((I, T-self.tInit)) # np.random.normal(mu, sigma, size=[I]) # co-integration noise
            #sigma = (max(self.kappa) / self.kappa).reshape(-1, 1) * self.fundamentalValueTrue[:, 1:] / 10000
            #nu = np.random.normal(0, sigma, size=[I, T-1]) # co-integration noise
            
            for stock in self.stocks:
                for t in range(self.tInit, T):
                    self.fundamentalValue[:, stock, t] = self.fundamentalValue[:, stock, t-1] + nu[:, t-self.tInit-1] + self.kappa * (self.fundamentalValueTrue[stock, t]*np.ones(I)-self.fundamentalValue[:, stock, t-1])
             
            self.priceDictionary = self.fetch.saved('priceDictionary')
            
            self.bestAskPrice[:] = self.priceDictionary[0, :, 1]
            self.bestBidPrice[:] = self.priceDictionary[0, :, 0]
                        
            self.reference_prices_order_book()
                        
            #####################
            ### RL variables ####
            #####################            
            
            self.load_Q_values()
            #self.volatilities = self.fetch.saved('volatilities').tolist()
            #self.volatilityReference[:] = self.fetch.saved('volatilityReference')
            
            self.stateForecasting[:, :, :self.tInitForecast] = self.fetch.saved('stateForecasting')[:, :, -self.tInitForecast:]
            self.actionForecasting[:, :, :self.tInit] = self.fetch.saved('actionForecasting')[:, :, -self.tInit:]
            self.stateTrading[:, :, :self.tInitmicro] = self.fetch.saved('stateTrading')[:, :, -self.tInitmicro:]
            self.actionTrading[:, :, :self.tInitmicro] = self.fetch.saved('actionTrading')[:, :, -self.tInitmicro:]
            self.actionTradingChosen[:, :, :self.tInitmicro] = self.fetch.saved('actionTradingChosen')[:, :, -self.tInitmicro:]
            #self.actionTradingEffective[:, :, :self.tInit] = self.fetch.saved('actionTradingEffective_end')
            self.statePricing[:, :, :self.tInitmicro] = self.fetch.saved('statePricing')[:, :, -self.tInitmicro:]
            self.actionPricing[:, :, :self.tInitmicro] = self.fetch.saved('actionPricing')[:, :, -self.tInitmicro:]
            
            #self.RewardTrading[:, :, :self.tInitmicro] = fetch.saved('RewardTrading')[:, :, -self.tInitmicro:]

            self.knowledgeWaste[:, :, :self.tInitmicro] = fetch.saved('knowledgeWaste')[:, :, -self.tInitmicro:]

            ####################
            # Wallet variables #
            ####################
            
            #VolumeOwned
            self.VolumeOwned[:, :, :self.tInitBinary] = fetch.saved('VolumeOwned')[:, :, -self.tInitBinary:]
            self.Aequity[:, :, :self.tInitBinary] = fetch.saved('Aequity')[:, :, -self.tInitBinary:]
            
            #AvailableVolume
            self.AvailableVolume[:, :, :self.tInitBinary] = fetch.saved('AvailableVolume')[:, :, -self.tInitBinary:]
            #self.AvailableVolume[:, :, :self.tInit] = self.VolumeOwned[:, :, :self.tInit]
            
            #Wallet
            self.Wallet[:, :self.tInitBinary] = fetch.saved('Wallet')[:, -self.tInitBinary:]
            
            #AvailableMargin
            self.AvailableMargin[:, :self.tInitBinary] = fetch.saved('AvailableMargin')[:, -self.tInitBinary:]
            #self.AvailableMargin[:, :self.tInit] = self.Wallet[:, :self.tInit]
            
            self.averageBuyPrice[:, :, :self.tInitBinary] = fetch.saved('averageBuyPrice')[:, :, -self.tInitBinary:]
            
            self.BalanceSheet[:, :, :self.tInitBinary] = fetch.saved('BalanceSheet')[:, :, -self.tInitmicro:]
            self.BalanceSheetBank[:, :self.tInitBinary] = fetch.saved('BalanceSheetBank')[:, -self.tInitmicro:]
            
            self.Forecast[:, :, :self.tInitForecast] = fetch.saved('Forecast')[:, :, -self.tInitForecast:]
            
            #Atotal
            #self.Atotal[:, :self.tInit] = fetch.saved('Atotal_end')
                            
        self.Atotal = self.Wallet + self.Aequity.sum(axis=1)   
        
        #breakpoint()
        #import os.path as path
        #path1 = path.abspath(path.join(__file__ ,"../.."))
        #path2 = '/'+fetch.DataFolderPath+f'/ex105'+f'/sim_0'+fetch.data_arrays_file()+'.npz'
        #self.Wallet2 = np.load('data_arrays.npz')['Wallet']
        #self.Price2 = np.load('data_arrays.npz')['Price']
        #self.actionForecasting2 = np.load('data_arrays.npz')['actionForecasting']
        #self.actionTrading2 = np.load('data_arrays.npz')['actionTrading']
        #self.QForecasting2 = np.load('data_arrays.npz')['QForecasting']
        
    def info(self, additional=""):
        
        """
        Prints the time step and stock.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        additional : str, optional
            More info to be displayed (default is None)

        Returns
        -------
        None
        """

        print(f'Processing time step {self.t} for stock {self.j}.' + additional)

    def create_stochastic_process(self, t0, baseline):

        T = self.T-t0

        ### Log-normal

        # Calculate parameters for the log-normal distribution
        #mu = np.log(1 + self.mu) - (self.sigma ** 2) / 2
        #sigma = self.sigma / np.sqrt(self.Ty)
    
        # Time step drift and volatility
        #drift = mu / self.Ty
        #vol = sigma

        # Initialize the process array
        #process = np.zeros((self.J, T))
        #process[:, 0] = baseline  # Initial value
    
        # Generate random factors
        #random_factors = np.random.normal(drift, vol, (self.J, T-1))
    
        # Calculate the stochastic process
        #process[:, 1:] = np.exp(random_factors)
        #process = np.cumprod(process, axis=1)
        
        ### Multiplicative (1+epsilon)
        
        # Calculate daily growth rate and volatility
        #daily_growth_rate = (self.mu+1)**(1/self.Ty) -1 #self.mu / self.Ty
        #daily_volatility = self.sigma / np.sqrt(self.Ty)
    
        # Initialize the process array
        #process = np.zeros((self.J, T))
        #process[:, 0] = baseline  # Initial value
    
        # Generate random factors
        #random_factors = np.random.normal(daily_growth_rate, daily_volatility, (self.J, T-1))
    
        # Calculate the simple multiplicative process
        #process[:, 1:] = 1 + random_factors
        #process = np.cumprod(process, axis=1)
        
        ### Multiplicative: Geometric Brownian Motion

        sigma = self.sigma / np.sqrt(self.Ty)

        # Derived daily growth rate for median growth rate
        #mu = np.log(1 + self.mu) / self.Ty + 0.5 * sigma**2

        # Derived daily growth rate for mean growth rate
        mu = np.log(1 + self.mu) / self.Ty

        # Initialize the process array
        process = np.zeros((self.J, T))
        process[:, 0] = baseline  # Initial value    
    
        random_factors = np.random.normal(size=(self.J, T-1))
    
        # Simulate the process
        for t in range(1, T):
            phi = (mu - 0.5 * sigma**2) + sigma * random_factors[:, t-1]
            process[:, t] = process[:, t-1] * np.exp(phi)
    
        return process

    def compress_variables(self):
    
        if not self.saveMicro:
    
            self.fundamentalValue = self.fundamentalValue[:, :, -self.tInit:]
            #self.fundamentalValue = self.fundamentalValue[:, :, -self.tInitForecast:]
            self.stateForecasting = self.stateForecasting[:, :, -self.tInitForecast:]
            self.Forecast = self.Forecast[:, :, -self.tInitForecast:]

    def load_Q_values(self):
        
        #Forecasting
        self.QForecasting[:, :self.tInitmicro] = self.fetch.saved('QForecasting')[:, -self.tInitmicro:]
        self.VForecasting[:, :self.tInitmicro] = self.fetch.saved('VForecasting')[:, -self.tInitmicro:]
                    
        #Trading
        self.QTrading[:, :self.tInitmicro] = self.fetch.saved('QTrading')[:, -self.tInitmicro:]
        self.VTrading[:, :self.tInitmicro] = self.fetch.saved('VTrading')[:, -self.tInitmicro:]
                            
        #Pricing
        self.QPricing[:, :self.tInitmicro] = self.fetch.saved('QPricing')[:, -self.tInitmicro:]
        self.VPricing[:, :self.tInitmicro] = self.fetch.saved('VPricing')[:, -self.tInitmicro:]
        
    def LSE_reference_stock(self, fetch):
    
        LSE = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)
        LSEstock = fetch.input_parameters()['LSEstock']
        self.PriceLSE = LSE[LSEstock].round(2)

        self.PriceLSEInit = self.PriceLSE.iloc[:self.tInit] #.values[::-1] #GL9

    def set_order_books(self):
    
         #Order Book column names
        self.OrderBookcolumns = ['Time', 'ID', 'Volume', 'Price', 'OrderCost', 'OrderFees', 'OrderType']

        self.OrderBookAs = [pd.DataFrame(columns=self.OrderBookcolumns),]*self.J #create J OrderBookAsks, one for each stock    
        self.OrderBookBs = [pd.DataFrame(columns=self.OrderBookcolumns),]*self.J #create J OrderBookBids, one for each stock
        
        self.AskOrderBookName = 'Asks'
        self.BidOrderBookName = 'Bids'
        
        [OrderBookAsks.index.rename(self.AskOrderBookName, inplace=True) for OrderBookAsks in self.OrderBookAs]
        [OrderBookBids.index.rename(self.BidOrderBookName, inplace=True) for OrderBookBids in self.OrderBookBs]

        for OrderBookAsks in self.OrderBookAs:
            OrderBookAsks.ID = OrderBookAsks.ID.astype(int)
            
        for OrderBookBids in self.OrderBookBs:
            OrderBookBids.ID = OrderBookBids.ID.astype(int)

        self.AskOrdersToClear = self.OrderBookAs[0].copy()
        self.BidOrdersToClear =  self.OrderBookBs[0].copy()

        self.bestAskPrice = np.zeros(self.J)        
        self.bestBidPrice = np.zeros(self.J)

        self.TimeAccuracy = max(6, round(np.log10(self.I)+1))

    ######################## Order Prices #########################
                        
    def Price_limit_order(self, traders, stock):

        self.LimitPrices[traders, stock, 0] = self.LastAuctionPriceReference[traders, stock]
        self.LimitPrices[traders, stock, 1] = self.LastAuctionPriceReference[traders, stock]
        
        self.LimitPrices[traders, stock, 2] = self.NextPriceReference[traders, stock]
        self.LimitPrices[traders, stock, 3] = self.NextPriceReference[traders, stock]

        self.LimitPrices[traders, stock] += self.PriceDeflection[traders, stock]

        self.LimitPrices[traders, stock] = self.LimitPrices[traders, stock].round(2)
        #print('Warning: correcting negative P_bid') if sum( Porder <= self.tick*np.ones(len(Porder))) > 0 else None
        self.LimitPrices[traders, stock] = np.maximum(self.LimitPrices[traders, stock], self.tick) # !!! Added artificially (in case Porder goes below tick)
        #breakpoint() if sum( Porder < 2*self.tick*np.ones(len(Porder))) > 0 else None

    #def price_deflection4lim(self, traders, stock):
    def price_deflection(self, traders, stock):

        #s = self.statePricing[traders, stock, self.tmicro, 0] #buying or selling
        #sDummy = np.sign((s==1)-.5)
        
        #g = self.gTilde[traders]*self.gSkewness[traders]*self.sDummy[traders]   

        #deflection = g * self.LastAuctionPriceReference[traders, stock]
  
        #self.PriceDeflection[traders, stock, 0] = deflection.round(2)
        #self.PriceDeflection[traders, stock, 1] = -(deflection/2).round(2)

        #deflection = g * self.NextPriceReference[traders, stock]

        #self.PriceDeflection[traders, stock, 2] = deflection.round(2)
        #self.PriceDeflection[traders, stock, 3] = -(deflection/2).round(2)

        gFavourite = self.gTildeFav[traders]*self.gSkewness[traders]*self.sDummy[traders]   
        gUnfavourite = self.gTildeUnfav[traders]*self.gSkewness[traders]*self.sDummy[traders] 
        
        self.PriceDeflection[traders, stock, 0] = self.LastAuctionPriceReference[traders, stock].copy()
        self.PriceDeflection[traders, stock, 1] = self.LastAuctionPriceReference[traders, stock].copy()
        
        self.PriceDeflection[traders, stock, 0] *= gFavourite
        self.PriceDeflection[traders, stock, 1] *= gUnfavourite

        self.PriceDeflection[traders, stock, 2] = self.NextPriceReference[traders, stock].copy()
        self.PriceDeflection[traders, stock, 3] = self.NextPriceReference[traders, stock].copy()

        self.PriceDeflection[traders, stock, 2] *= gFavourite
        self.PriceDeflection[traders, stock, 3] *= gUnfavourite

        self.PriceDeflection[traders, stock] = self.PriceDeflection[traders, stock].round(2)

    ######################## Order Volumes #########################

    def Volume_ask(self, traders, stock):
        
        #Vask = np.maximum( self.AvailableVolume[traders, stock, self.tBinary], self.AvailableVolume[traders, stock, self.tInitBinary-1] ) * self.rhoAsk[traders]
        
        Vask = np.maximum( self.InitialVolumeOwned, self.AvailableVolume[traders, stock, self.tBinary]* self.rhoAsk[traders] )
        Vask = np.minimum(Vask, self.AvailableVolume[traders, stock, self.tBinary])
        
        return ( Vask ).astype(int)
        
    def Volume_bid(self, traders, stock):
                                       
        UnitaryTotalOrderCost = self.OrderPrices[traders, stock]*(1+ 2*self.b) #counting twice, also for selling (although price may be higher, and estimation may not be enough)
        Vbid = ( self.CashToInvest[traders] / UnitaryTotalOrderCost ).astype(int)
          
        return Vbid

    ######################## IDs ########################
      
    def trading_IDs(self, stock):
                
        self.trading_IDs_now(stock)        
        self.trading_IDs_limit(stock)
        self.trading_IDs_market(stock)

    def trading_IDs_now(self, stock):

        self.AskCondition = (self.actionTrading[self.traders, stock, self.tmicro] > 1) #.astype(bool)
        self.BidCondition = (self.actionTrading[self.traders, stock, self.tmicro] < 1) #.astype(bool)

        self.TradingNowCondition = self.AskCondition+self.BidCondition
        self.TradingNow = self.tradersArray[self.TradingNowCondition]
        self.NotTradingNow = self.tradersArray[(1-self.TradingNowCondition).astype(bool)]
        self.AskTraders = self.tradersArray[self.AskCondition]        
        self.BidTraders = self.tradersArray[self.BidCondition]

    def trading_IDs_limit(self, stock):
        #breakpoint()
        self.LimitCondition = (self.actionPricing[self.traders, stock, self.tmicro] < self.NactionsPricingLimit) #.astype(bool) #may contain non-trading traders
        self.LimitTraders = self.tradersArray[self.LimitCondition*self.TradingNowCondition]
        self.AskLimitCondition = self.AskCondition*self.LimitCondition        
        self.AskLimitTraders = self.tradersArray[self.AskLimitCondition]
        self.BidLimitCondition = self.BidCondition*self.LimitCondition                
        self.BidLimitTraders = self.tradersArray[self.BidLimitCondition]

    def trading_IDs_market(self, stock):
    
        self.MarketCondition = (1-self.LimitCondition)*self.TradingNowCondition
        self.MarketTraders = self.tradersArray[self.MarketCondition.astype(bool)]
        self.AskMarketCondition = self.AskCondition*self.MarketCondition
        self.AskMarketTraders = self.tradersArray[self.AskMarketCondition.astype(bool)]
        self.BidMarketCondition = self.BidCondition*self.MarketCondition
        self.BidMarketTraders = self.tradersArray[self.BidMarketCondition.astype(bool)] 

    ######################## Validate trading decisions ########################

    def cash_to_invest(self):
        
        rhoBid = self.rhoBid
            
        self.CashToInvest = np.minimum( self.AvailableMargin[self.traders, self.tBinary], rhoBid*(self.AvailableMargin[self.traders, self.tBinary]+self.Aequity[self.traders, :, self.tBinary].sum(axis=1)) )

    def order_costs_and_fees(self, stock, traders):
    
        self.OrderCosts[traders, stock] = self.OrderVolumes[traders, stock]*self.OrderPrices[traders, stock]
        self.OrderFees[traders, stock] = self.OrderVolumes[traders, stock]*self.OrderPrices[traders, stock]*self.b 

    def check_bid_orders(self, traders, stock): #, orderType):

        self.OrderVolumes[traders, stock] = self.Volume_bid(traders, stock)
        VolumeCondition = (self.OrderVolumes[traders, stock] < 1)
        #print(f'Foreseen Total Cost: deleting {round(sum(1-VolumeCondition)/len(traders)*100, 2)} % bid {orderType} orders (Volume condition)') if sum(1-VolumeCondition) > 0 else None

        self.order_costs_and_fees(stock, traders)   
        ForeseenTotalCost = self.OrderFees[traders, stock] + self.OrderCosts[traders, stock]
        AvailableMarginCondition = (self.AvailableMargin[traders, self.tBinary] < ForeseenTotalCost)
        #print('Foreseen Total Cost: deleting orders (Margin condition)') if sum(1-AvailableMarginCondition) > 0 else None

        invalidOrdersCondition = (AvailableMarginCondition+VolumeCondition).astype(bool)
        invalidOrdersTraders = traders[invalidOrdersCondition]
        
        self.actionTrading[invalidOrdersTraders, stock, self.tmicro] = 1 #turn to 1 (=pass) where either condition is verified 
        self.OrderPrices[invalidOrdersTraders, stock] = 0 
        self.OrderVolumes[invalidOrdersTraders, stock] = 0 
        self.order_costs_and_fees(stock, invalidOrdersTraders)
                        
    def check_ask_orders(self, traders, stock):

        self.OrderVolumes[traders, stock] = self.Volume_ask(traders, stock)
        VolumeCondition = (self.OrderVolumes[traders, stock] < 1) #only useful if rhoAsk < 1, as 'ask' action is only available when volume owned is > 0 (see action_trading)
        #print(f'Foreseen Total Cost: deleting {round(sum(1-VolumeCondition)/len(traders)*100, 2)} % bid {orderType} orders (Volume condition)') if sum(1-VolumeCondition) > 0 else None

        self.OrderFees[traders, stock] = self.OrderVolumes[traders, stock]*self.OrderPrices[traders, stock]*self.b    
        ForeseenTotalCost = self.OrderFees[traders, stock]
        AvailableMarginCondition = (self.AvailableMargin[traders, self.tBinary] < ForeseenTotalCost)
        #print('Foreseen Total Cost: deleting orders (Margin condition)') if sum(1-AvailableMarginCondition) > 0 else None
        
        invalidOrdersCondition = (AvailableMarginCondition+VolumeCondition).astype(bool)
        invalidOrdersTraders = traders[invalidOrdersCondition]
        
        self.actionTrading[invalidOrdersTraders, stock, self.tmicro] = 1 #turn to 1 (=pass) where either condition is verified 
        self.OrderPrices[invalidOrdersTraders, stock] = 0 
        self.OrderVolumes[invalidOrdersTraders, stock] = 0 
        self.OrderFees[invalidOrdersTraders, stock] = 0 
        
    def validate_trading_decisions(self, stock):
        
        self.trading_IDs(stock)
        
        self.cash_to_invest()
        
        self.validate_prices(stock)
        self.validate_volumes(stock)
        breakpoint() if ((self.actionTrading[self.traders, stock, self.tmicro]==1)*(self.OrderVolumes[self.traders, stock]>0)).sum() > 0 else None
        self.validate_charges(stock)
        
        self.TradingFreeChoice = ( self.actionTradingChosen[self.traders, stock, self.tmicro] == self.actionTrading[self.traders, stock, self.tmicro] ) #not considering interaction/active decision

    def validate_prices(self, stock):
    
        #Check Limit Bid Orders
        actionPricing = self.actionPricing[self.BidLimitTraders, stock, self.tmicro]
        self.OrderPrices[self.BidLimitTraders, stock] = self.LimitPrices[self.BidLimitTraders, stock, actionPricing] #self.Price_limit_order(self.BidLimitTraders, stock, self.actionPricing[self.BidLimitTraders, stock, self.tmicro])             
        self.check_bid_orders(self.BidLimitTraders, stock) #, 'limit')

        #Limit Orders
        actionPricing = self.actionPricing[self.AskLimitTraders, stock, self.tmicro]
        self.OrderPrices[self.AskLimitTraders, stock] = self.LimitPrices[self.AskLimitTraders, stock, actionPricing] #self.Price_limit_order(self.AskLimitTraders, stock, self.actionPricing[self.AskLimitTraders, stock, self.tmicro])
        self.check_ask_orders(self.AskLimitTraders, stock)
        
        self.trading_IDs_now(stock)        
        self.trading_IDs_limit(stock)
        
        PriceLimit = self.OrderPrices[self.LimitTraders, stock]
        
        #Check and validate market Bid Orders

        if self.phase == 'pretraining':
           PbidMarket = self.PriceLSE.iloc[self.t]+self.tick
        else:
           PbidMarket = max(PriceLimit)+self.tick if PriceLimit.tolist() else self.Price[stock, self.t-1]+self.tick
                
        self.OrderPrices[self.BidMarketTraders, stock] = PbidMarket
        self.check_bid_orders(self.BidMarketTraders, stock) #, 'market')
        
        #breakpoint() if sum( self.OrderPrices[self.BidLimitTraders, stock] < 2*self.tick*np.ones(len(self.BidLimitTraders)) )> 0 else None
        
        #Ask Market Orders        
        if self.phase == 'pretraining':
           PaskMarket = self.PriceLSE.iloc[self.t]-self.tick
        else:
           PaskMarket = min(PriceLimit)-self.tick if PriceLimit.tolist() else self.Price[stock, self.t-1]-self.tick
           PaskMarket = np.maximum(PaskMarket, self.tick)

        self.OrderPrices[self.AskMarketTraders, stock] = PaskMarket
        self.check_ask_orders(self.AskMarketTraders, stock)

        self.trading_IDs_now(stock)
        self.trading_IDs_market(stock)
        
        self.OrderPrices[self.traders, stock] = self.OrderPrices[self.traders, stock].round(2)

    def validate_volumes(self, stock):
    
        #Volumes
        self.OrderVolumes[self.AskTraders, stock] = self.Volume_ask(self.AskTraders, stock)
        self.OrderVolumes[self.BidTraders, stock] = self.Volume_bid(self.BidTraders, stock)

        self.AvailableVolume[self.AskTraders, stock, self.tBinary] -= self.OrderVolumes[self.AskTraders, stock]

    def validate_charges(self, stock):
    
        #Order costs        
        self.OrderCosts[self.BidTraders, stock] = self.OrderVolumes[self.BidTraders, stock]*self.OrderPrices[self.BidTraders, stock]
        self.OrderCosts[self.BidTraders, stock] = self.OrderCosts[self.BidTraders, stock].round(2)
        self.OrderFees[self.TradingNow, stock] = self.OrderVolumes[self.TradingNow, stock]*self.OrderPrices[self.TradingNow, stock]*self.b
        self.OrderFees[self.TradingNow, stock] = self.OrderFees[self.TradingNow, stock].round(2)
        ForeseenTotalCost = self.OrderFees[self.traders, stock] + self.OrderCosts[self.traders, stock]

        self.AvailableMargin[self.TradingNow, self.tBinary] -= ForeseenTotalCost[self.TradingNow]

    #####################################
    ######### Reference Prices ##########
    #####################################

    def reference_prices_order_book(self):

        for stock in self.stocks:
            
            self.best_ask_price(stock), self.best_bid_price(stock)
            self.priceDictionary[:, stock, 0] = max( self.bestBidPrice[stock], self.tick )
            self.priceDictionary[:, stock, 1] = max( self.bestAskPrice[stock], self.tick )
                                      
        return
        
    def print_ref(self, stock):
        
        print(self.Price[stock, self.t-1]) if self.t == self.tInit else None
        print(self.priceDictionary[0, stock]) if self.t == self.tInit else None

    def print_ref2(self, stock):

        print(self.Price[stock, self.t-1]) if self.t == self.tInit else None   
        print('AU: ', self.priceDictionary[0, stock, 0], self.priceDictionary[0, stock, 1], self.priceDictionary[0, stock, 2], self.priceDictionary[0, stock, 3]) if self.t == self.tInit else None
        
    def best_ask_price(self, stock):
        
        OrderBookAsks = self.OrderBookAs[stock]
        
        if len(OrderBookAsks[OrderBookAsks['OrderType']=='Limit']) > 0:
            self.bestAskPrice[stock] = OrderBookAsks[OrderBookAsks['OrderType']=='Limit'].Price.min()
            #self.bestAskPrice[stock] = max( OrderBookAsks[OrderBookAsks['OrderType']=='Limit'].Price.min(), self.Price[stock, self.t]+self.tick)
        else:
            self.bestAskPrice[stock] = max(self.bestAskPrice[stock], self.Price[stock, self.t]+self.tick)

    def best_bid_price(self, stock):
        
        OrderBookBids = self.OrderBookBs[stock]
        
        if len(OrderBookBids[OrderBookBids['OrderType']=='Limit']) > 0:
            self.bestBidPrice[stock] = OrderBookBids[OrderBookBids['OrderType']=='Limit'].Price.max()
            #self.bestBidPrice[stock] = min(OrderBookBids[OrderBookBids['OrderType']=='Limit'].Price.max(), self.Price[stock, self.t]-self.tick)
        else:
            self.bestBidPrice[stock] = min(self.bestBidPrice[stock], self.Price[stock, self.t]-self.tick)
            #print('bestBidPrice (no limit order)', self.bestBidPrice[stock], 'vs', self.Price[stock, self.t]-self.tick)

    ##########################################################
    ################## Orders on OrderBook ###################
    ##########################################################
      
    def orders_boolean(self, TradingType, OrderType='Trading'):
    
        '''
        Tradingtype: 'Asks' or 'Bids'
        OrderType: 'Trading', 'Market' or 'Limit'
        '''
        
        OrderBook = self.OrderBookAsks if TradingType == 'Asks' else self.OrderBookBids if TradingType == 'Bids' else None
        OrdersBoolean = len( OrderBook ) > 0 if OrderType == 'Trading' else len( OrderBook[ OrderBook.OrderType == OrderType] ) > 0
            
        return OrdersBoolean

    def add_new_orders(self, OrderBook, traders, OrderType):
        
        OrderBookName = OrderBook.index.name
        
        nTraders = len(traders)
        
        Time = self.t+ np.random.rand(nTraders)*.5 #differentiate time within time step            
        Time += .5*(OrderType == 'Limit')

        Volumes = self.OrderVolumes[traders, self.stock]
        Prices = self.OrderPrices[traders, self.stock]
        OrderCosts = self.OrderCosts[traders, self.stock]
        OrderFees = self.OrderFees[traders, self.stock]
        OrderType = [OrderType]*nTraders        
        
        OrderBookdict = dict(zip(self.OrderBookcolumns, [Time, traders, Volumes, Prices, OrderCosts, OrderFees, OrderType]))        
        NewOrders = pd.DataFrame(OrderBookdict)
        
        #Update Order Book
        OrderBook = pd.concat([OrderBook, NewOrders], ignore_index=True) if len(OrderBook)>0 else NewOrders
        OrderBook.index.name = OrderBookName
        
        breakpoint() if sum(OrderBook.Volume < 1) > 0 else None        
        
        return OrderBook
             
    def new_orders(self):
    
        self.trading_IDs(self.stock)
        
        ###### New SELL market orders ######
        self.OrderBookAsks = self.add_new_orders(self.OrderBookAsks, self.AskMarketTraders, 'Market') if self.AskMarketCondition.sum() > 0 else self.OrderBookAsks
        
        ###### New BUY market orders ######
        self.OrderBookBids = self.add_new_orders(self.OrderBookBids, self.BidMarketTraders, 'Market') if self.BidMarketCondition.sum() > 0 else self.OrderBookBids

        ###### New SELL limit orders ######
        self.OrderBookAsks = self.add_new_orders(self.OrderBookAsks, self.AskLimitTraders, 'Limit') if self.AskLimitCondition.sum() > 0 else self.OrderBookAsks
        
        ###### New BUY limit orders ######
        self.OrderBookBids = self.add_new_orders(self.OrderBookBids, self.BidLimitTraders, 'Limit') if self.BidLimitCondition.sum() > 0 else self.OrderBookBids
        
        self.sort_order_book(self.OrderBookAsks) #Sort Order Book
        self.sort_order_book(self.OrderBookBids) #Sort Order Book
        
        #self.OrderBookAs[self.stock] = self.OrderBookAsks
        #self.OrderBookBs[self.stock] = self.OrderBookBids

    def sort_order_book(self, OrderBook):
    
        if OrderBook.index.name == self.AskOrderBookName:
            
            OrderBook.sort_values(by=['Price', 'Time'], ascending=[True, True], inplace=True)
        
        elif OrderBook.index.name == self.BidOrderBookName:
            
            OrderBook.sort_values(by=['Price', 'Time'], ascending=[False, True], inplace=True)
                        
   ###########################################################################
   ################################ AUCTION  #################################
   ########################################################################### 
     
    def auction(self, stock):
        
        self.stock = stock
        
        self.OrderBookAsks = self.OrderBookAs[stock]
        self.OrderBookBids = self.OrderBookBs[stock]
                
        self.new_orders()
        
        #Diagnostic purposes
        #self.OldOBA = self.OrderBookAsks
        #self.OldOBB = self.OrderBookBids
        
        if self.phase in ['pretraining_nobias', 'training', 'testing', 'calibrating']:
        
            self.execute_auction()
            self.test_clearing()    
            
        elif self.phase == 'pretraining':
        
            self.clear_orders()
                    
        self.returns[self.stock, self.t-1] = np.log( self.Price[self.stock, self.t] ) - np.log( self.Price[self.stock, self.t-1] )
        self.BidAskSpread[self.stock, self.t] = self.OrderBookAsks.Price.iloc[0] - self.OrderBookBids.Price.iloc[0] if self.orders_boolean('Asks', 'Limit') and self.orders_boolean('Bids', 'Limit') else 0
                
        self.test_bidaskspread()
        
        ### Update list of order books ###
        self.OrderBookAs[self.stock] = self.OrderBookAsks
        self.OrderBookBs[self.stock] = self.OrderBookBids
        
        #print(self.Price[self.stock, self.t-10:self.t]) if np.mod(self.t, 1000) == tInit-1 else None
        
        if self.verbose:
            print(self.Price[self.stock, self.t-10:self.t]) if np.mod(self.t, 200) == 100 else None
            print('zero returns %', ((self.returns[stock, self.t-self.tInit:self.t] == 0).mean()*100).round(2)) if np.mod(self.t, 200) == 100 else None
            print('min / max Q', self.QTrading[:, self.tmicro].min(), self.QTrading[:, self.tmicro].max()) if np.mod(self.t, 200) == 100 else None
        
        return
          
    def execute_auction(self):
        
        #print(self.t)
        #print(self.OrderBookAsks)
        #print()
        #print()
        #print(self.OrderBookBids)
        
        if self.orders_boolean('Asks') and self.orders_boolean('Bids'): #at least someone selling, someone buying = both order books are not empty
        
            self.supply_demand_curve() if self.OrderBookAsks.Price.iloc[0] <= self.OrderBookBids.Price.iloc[0] else self.no_intersection() #if best ask and best bid are compatible
                           
        else: #If at least one order book is empty
            
            self.empty_OrderBooks()
        
        self.Volume[self.stock, self.t] = self.VolumeToClear  
        self.resolvedTrading[self.traders, self.stock, self.tmicro] = self.resolvedAsk[self.traders, self.stock, self.tmicro]+self.resolvedBid[self.traders, self.stock, self.tmicro]
        
        #breakpoint() if sum(self.AvailableMargin[:, self.t] > self.Wallet[:, self.t]) else None
                            	    
    def supply_demand_curve(self):
    
        if self.orders_boolean('Asks', 'Limit') and self.orders_boolean('Bids', 'Limit'):
            
            self.cumulative_volumes() #compute cumulative distribution of volumes to find intersection
            self.find_intersection() #find intersection                

            breakpoint() if self.VolumeToClear < 1 else None

        elif self.orders_boolean('Asks', 'Limit') and not self.orders_boolean('Bids', 'Limit'):
        
            
            self.Price[self.stock, self.t] = self.OrderBookAsks[self.OrderBookAsks.OrderType=='Limit'].Price.max()
            
            cumulativeAsk = self.OrderBookAsks.Volume.sum()
            cumulativeBid = self.OrderBookBids.Volume.sum()
            self.VolumeToClear = min( cumulativeAsk, cumulativeBid ) #if self.orders_boolean('Bids', 'Market') else 0

            message = f'Only Limit Ask orders; Volume to Clear: {self.VolumeToClear}'
            self.log_message(message)
            
        elif self.orders_boolean('Bids', 'Limit') and not self.orders_boolean('Asks', 'Limit'):
                    
            self.Price[self.stock, self.t] = self.OrderBookBids[self.OrderBookBids.OrderType=='Limit'].Price.min()

            cumulativeAsk = self.OrderBookAsks.Volume.sum()
            cumulativeBid = self.OrderBookBids.Volume.sum()
            self.VolumeToClear = min( cumulativeAsk, cumulativeBid ) #if self.orders_boolean('Asks', 'Market') else 0

            message = f'Only Limit Bid orders; Volume to Clear: {self.VolumeToClear}'
            self.log_message(message)

        else: 
                
            self.Price[self.stock, self.t] = self.Price[self.stock, self.t-1]
            
            cumulativeAsk = self.OrderBookAsks.Volume.sum()
            cumulativeBid = self.OrderBookBids.Volume.sum()
            self.VolumeToClear = min( cumulativeAsk, cumulativeBid )

            message = f'Only Market orders; Volume to Clear: {self.VolumeToClear}'
            self.log_message(message)

        breakpoint() if self.Price[self.stock, self.t] < self.tick else None

        self.volume_clearing_sellers() ### Volume distribution to sellers    
        self.volume_clearing_buyers() ### Volume distribution to buyers

        #print(self.Price[self.stock, self.t])
        #breakpoint()
        #print('\n')
        #print(self.t)
        #print('\n')
        #print(self.AskOrdersToClear)
        #print(self.BidOrdersToClear)
        #print('\n')

        self.update_portfolio_sellers()
        self.update_portfolio_buyers()         
            
    def cumulative_volumes(self):
        
        Ask = self.OrderBookAsks.groupby('Price').Volume.sum()
        Bid = self.OrderBookBids.groupby('Price').Volume.sum()

        AskPrices = Ask.index.values
        BidPrices = Bid.index.values

        ### Joint price range for cumulative distribution of volumes
        self.prices = np.arange(min(BidPrices[0], AskPrices[0])-self.safeMarginPrices, max(AskPrices[-1], BidPrices[-1])+self.safeMarginPrices, self.tick).round(2)
        
        self.AskVolumesCumulative, self.BidVolumesCumulative = np.zeros(len(self.prices)).astype(int), np.zeros(len(self.prices)).astype(int)

        for price in AskPrices:

            self.AskVolumesCumulative[(price==self.prices)] = Ask.loc[price]

        self.AskVolumesCumulative = self.AskVolumesCumulative.cumsum()

        for price in BidPrices:

            self.BidVolumesCumulative[(price==self.prices)] = Bid.loc[price]

        self.BidVolumesCumulative = self.BidVolumesCumulative[::-1].cumsum()[::-1]
    
    def find_intersection(self):
    
        MinimalVolume = np.minimum(self.AskVolumesCumulative, self.BidVolumesCumulative)
        self.VolumeToClear = np.max(MinimalVolume)
        
        indeces = np.nonzero(MinimalVolume == self.VolumeToClear)[0]

        if len(indeces) > 1:

            imbalance = abs( self.BidVolumesCumulative[indeces]-self.AskVolumesCumulative[indeces] )
            
            indecesImbalance = np.nonzero(imbalance == np.min(imbalance))[0]
            indeces = indeces[indecesImbalance]
            
            if len(indeces) > 1:
                
                latestPriceProximity = abs( self.prices[ indeces ] - self.Price[self.stock, self.t-1] )
                indecesPriceProximity = np.nonzero(latestPriceProximity == np.min(latestPriceProximity))[0]
                indeces = indeces[indecesPriceProximity]
                #print('proximity') if self.prices[ indeces[0] ]==self.Price[self.stock, self.t-1] else None
                
                if len(indeces) > 1:

                    indeces = np.random.choice(indeces)
                    
                    message = f'random criterion at price intersection, time {self.t}'
                    self.log_message(message)
                    
        self.intersection = indeces

        self.Price[self.stock, self.t] = self.prices[ self.intersection ]
        #self.Price[self.stock, self.t] = max(self.Price[self.stock, self.t], self.tick)
    
        ### Diagnostics for auction
        #self.plot_diagnostics()
      
    def no_intersection(self):
        
        message = f'no intersection at time {self.t}, Price {self.Price[self.stock, self.t-1]}'
        self.log_message(message)
        
        self.Price[self.stock, self.t] = ( self.OrderBookBids.Price.max() + self.OrderBookAsks.Price.min() ) / 2 #self.Price[self.stock, self.t-1]
        self.VolumeToClear = 0

    def clear_orders(self):
        
        '''
        for pretraining only
        '''
        
        self.Price[self.stock, self.t] = self.PriceLSE.iloc[self.t]

        AskCondition = self.OrderBookAsks.Price <= self.Price[self.stock, self.t]
        AskVolume = self.OrderBookAsks[AskCondition].Volume.sum() if AskCondition.sum() > 0 else 0        
        BidCondition = self.OrderBookBids.Price >= self.Price[self.stock, self.t]
        BidVolume = self.OrderBookBids[BidCondition].Volume.sum() if BidCondition.sum() > 0 else 0
        
        self.VolumeToClear = AskVolume+BidVolume
        self.Volume[self.stock, self.t] = AskVolume+BidVolume
              
        (self.volume_clearing_sellers(), self.update_portfolio_sellers()) if AskVolume > 0 else None
        (self.volume_clearing_buyers(), self.update_portfolio_buyers()) if BidVolume > 0 else None
       
        self.resolvedTrading[:, self.stock, self.t] = self.resolvedAsk[:, self.stock, self.t]+self.resolvedBid[:, self.stock, self.t]

    def label_volume_to_clear(self, condition, OrderBook):
    
        #OrderBook['MatchVolume'] = condition * OrderBook.Volume
        #OrderBook['ToClear'] = condition * ( OrderBook.MatchVolume.cumsum().apply(lambda x: x <= self.VolumeToClear ) ) if self.phase in ['training', 'testing', 'calibrating'] else condition if self.phase == 'pretraining' else None
        #OrderBook.drop('MatchVolume', axis=1, inplace=True)
        
        #OrderBook['ToClear'] = condition * ( (condition * OrderBook.Volume).cumsum().apply(lambda x: x <= self.VolumeToClear ) ) if self.phase in ['training', 'testing', 'calibrating'] else condition if self.phase == 'pretraining' else None
                
        OrderBook['ToClear'] = condition * ( (condition * OrderBook.Volume).cumsum() <= self.VolumeToClear ) if self.phase in ['pretraining_nobias', 'training', 'testing', 'calibrating'] else condition if self.phase == 'pretraining' else None
        
        return OrderBook
       
    def volume_clearing(self, OrderBook, Condition):
        
        OrderBook = self.label_volume_to_clear(Condition, OrderBook)
        
        OrdersToClear = OrderBook.loc[ OrderBook.ToClear == True ].copy()
        
        #OrderBook.drop('ToClear', axis=1, inplace=True)
        OrderBookcleared = OrderBook.drop( OrdersToClear.index ).copy() 

        if self.phase in ['pretraining_nobias', 'training', 'testing', 'calibrating']:
            OrdersToClear, OrderBookcleared = self.volume_clearing_leftover( OrdersToClear, OrderBookcleared )
                   
        OrderBookcleared.index.rename(OrderBook.index.name, inplace=True)
        
        return OrderBookcleared, OrdersToClear

    def volume_clearing_leftover(self, OrdersToClear, OrderBookcleared ):
    
        LeftOverVolumesToClear = self.VolumeToClear - OrdersToClear.Volume.sum()
               
        if LeftOverVolumesToClear > 0:
            OrdersToClear, OrderBookcleared = self.leftover_volumes(OrdersToClear, OrderBookcleared, LeftOverVolumesToClear)  
        
        return OrdersToClear, OrderBookcleared 
    
    def leftover_volumes(self, OrdersToClear, OrderBookCleared, LeftOverVolumesToClear):
        
        '''
        left-over volume, take fraction of total order from next best order
        '''
        
        #OrderBookCleared.ID = OrderBookCleared.ID.astype(int)
        
        LeftOverToClear = OrderBookCleared.iloc[0].copy() 
        LeftOverToClear.Volume = LeftOverVolumesToClear
        LeftOverToClear.OrderCost = LeftOverToClear.Volume * LeftOverToClear.Price
        LeftOverToClear.OrderFees = LeftOverToClear.OrderCost * self.b #if b, not bFlat
        LeftOverToClear.OrderCost *= (OrderBookCleared.index.name == self.BidOrderBookName) #send to 0 ask orders (gaining, not spending)
        LeftOverToClear['ToClear'] = True
        
        LeftOverToClear = LeftOverToClear.to_frame().T
        OrdersToClear = pd.concat([OrdersToClear, LeftOverToClear], ignore_index=True)
        OrdersToClear.ID = OrdersToClear.ID.astype(int)
        
        index = LeftOverToClear.index[0]
        columns = ['Volume', 'OrderCost', 'OrderFees']
        OrderBookCleared.loc[index, columns] -= LeftOverToClear.loc[index, columns]
              
        return OrdersToClear, OrderBookCleared

    def volume_clearing_sellers(self):
        
        AskCondition = ( self.OrderBookAsks.Price <= self.Price[self.stock, self.t] )
        self.OrderBookAsks, self.AskOrdersToClear = self.volume_clearing( self.OrderBookAsks, AskCondition )
    
    def volume_clearing_buyers(self):
            
        BidCondition = ( self.OrderBookBids.Price >= self.Price[self.stock, self.t] )              
        self.OrderBookBids, self.BidOrdersToClear = self.volume_clearing( self.OrderBookBids, BidCondition )

    def isclose(self, ID):
    
        smallValuesBinaryCheck = np.isclose(self.Wallet[ID, self.tBinary], 0, atol=self.tick/2) #can use "np.isclose"
        self.Wallet[ID, self.tBinary] *= (1-smallValuesBinaryCheck) #put 0 for condition True; otherwise keep value (*1)
        self.Wallet[ID, self.tBinary] = self.Wallet[ID, self.tBinary].round(2)
                               
    def update_portfolio_sellers(self):
                    
        ID = self.AskOrdersToClear.ID.values
        
        #self.actionTradingEffective[ID, self.stock, self.t] = 2
        
        Volume = self.AskOrdersToClear.Volume
        Price = self.Price[self.stock, self.t]
        
        self.VolumeOwned[ID, self.stock, self.tBinary] -= Volume #may be lower quantity than issued order, but remaining stocks are still on the order book = not available   
      
        self.BalanceSheet[ID, self.stock, self.tmicro, 0] = Volume * Price * self.b
        self.BalanceSheet[ID, self.stock, self.tmicro, 1] = Volume * Price

        self.BalanceSheetFees[ID, self.stock, self.tmicro] += Volume * Price * self.b #broker fees
               
        self.DeltaWalletTrading[ID] = self.BalanceSheet[ID, self.stock, self.tmicro, 1]-self.BalanceSheet[ID, self.stock, self.tmicro, 0]
        self.Wallet[ID, self.tBinary] += self.DeltaWalletTrading[ID]
                
        self.isclose(ID)

        self.AvailableMargin[ID, self.tBinary] = self.Wallet[ID, self.tBinary]
        
        if ID[-1] in self.OrderBookAsks.ID.values:
            self.AvailableMargin[ID[-1], self.tBinary] -= self.OrderBookAsks[self.OrderBookAsks.ID==ID[-1]].iloc[0].OrderFees  
        self.AvailableMargin[ID, self.tBinary] = self.AvailableMargin[ID, self.tBinary].round(2)

        VolumeCondition = (self.VolumeOwned[ID, self.stock, self.tBinary]>0) #.astype(bool) #ask order may have been filled only partially  
        IDclosed = ID[(1-VolumeCondition).astype(bool)] #Partial selling is possible when orders are not fully cleared
        IDpartial = ID[VolumeCondition] #Partial selling is possible when orders are not fully cleared
        self.resolvedAsk[ID, self.stock, self.tmicro] = 1
        
        self.averageBuyPrice[IDclosed, self.stock, self.tBinary] = 0
        
    def update_portfolio_buyers(self):
        
        #self.BidOrdersToClear.sort_values(by='ID', inplace=True)
        ID = self.BidOrdersToClear.ID.values
        #self.actionTradingEffective[ID, self.stock, self.t] = 0
        #breakpoint()

        breakpoint() if sum(self.BidOrdersToClear.Volume <= 0) > 0 else None
        
        #breakpoint() if self.t == 515 else None
        
        Volume = self.BidOrdersToClear.Volume
        Price = self.Price[self.stock, self.t]
        
        self.VolumeOwned[ID, self.stock, self.tBinary] += Volume
        self.AvailableVolume[ID, self.stock, self.tBinary] +=  Volume
        
        self.BalanceSheet[ID, self.stock, self.tmicro, 0] = Volume * Price #order costs
        self.BalanceSheet[ID, self.stock, self.tmicro, 0] += Volume * Price * self.b #broker fees
        
        self.BalanceSheetFees[ID, self.stock, self.tmicro] += Volume * Price * self.b #broker fees
        
        self.DeltaWalletTrading[ID] -= self.BalanceSheet[ID, self.stock, self.tmicro, 0] #self.BalanceSheet[ID, self.stock, self.tmicro, 1]
        self.Wallet[ID, self.tBinary] += self.DeltaWalletTrading[ID]   
        
        self.isclose(ID)
        
        self.AvailableMargin[ID, self.tBinary] = np.copy( self.Wallet[ID, self.tBinary] )

        if ID[-1] in self.OrderBookBids.ID.values:
            self.AvailableMargin[ID[-1], self.tBinary] -= (self.OrderBookBids[self.OrderBookBids.ID==ID[-1]].iloc[0].OrderFees+self.OrderBookBids[self.OrderBookBids.ID==ID[-1]].iloc[0].OrderCost) 

        self.AvailableMargin[ID, self.tBinary] = self.AvailableMargin[ID, self.tBinary].round(2)
               
        VolumeCondition = self.VolumeOwned[ID, self.stock, self.tBinary-1]<1
        IDopening = ID[VolumeCondition]
        
        self.resolvedBid[ID, self.stock, self.tmicro] = 1
        
        self.averageBuyPrice[ID, self.stock, self.tBinary] = self.averageBuyPrice[ID, self.stock, self.tBinary-1]*self.VolumeOwned[ID, self.stock, self.tBinary-1] + Price*Volume
        self.averageBuyPrice[ID, self.stock, self.tBinary] /= self.VolumeOwned[ID, self.stock, self.tBinary]
                
        self.test_consistency(ID) if sum(self.Wallet[ID, self.tBinary] < 0) > 0 else None

    def empty_OrderBooks(self):
        
        message = f'empty order book; deciding: {self.TradingDecisionMakersBinary.sum()}'
        self.log_message(message)
        
        self.Price[self.stock, self.t] = self.OrderBookAsks.Price.min() if self.orders_boolean('Asks') else self.OrderBookBids.Price.max() if self.orders_boolean('Bids') else self.Price[self.stock, self.t-1] if not self.orders_boolean('Asks') and not self.orders_boolean('Bids') else None
        
        #breakpoint() if np.isnan(self.Price[self.stock, self.t]) else None
        
        self.VolumeToClear = 0


    #############################################
    ################# CANCELLING ################
    #############################################
    
    # !!! If multiple orders per agent are allowed, there may be multiple buy or sell orders on the order books from the same trader
    
    def cancel_orders(self, stock):

        self.OrderBookAsks, self.OrderBookBids = self.OrderBookAs[stock], self.OrderBookBs[stock]
                
        self.AvailableMargin[self.OrderBookAsks.ID.values, self.tBinary] += self.OrderBookAsks.OrderCost + self.OrderBookAsks.OrderFees       
        self.AvailableMargin[self.OrderBookBids.ID.values, self.tBinary] += self.OrderBookBids.OrderCost + self.OrderBookBids.OrderFees
        self.AvailableMargin[:, self.tBinary] = self.AvailableMargin[:, self.tBinary].round(2)
        
        #breakpoint() if sum( np.isclose(self.AvailableMargin[:, self.t], self.Wallet[:, self.t], atol=self.tick) ) < self.I else None
        if sum( np.isclose(self.AvailableMargin[:, self.tBinary], self.Wallet[:, self.tBinary], atol=self.tick) ) < self.Ntraders:
            
            message = 'Warning: Amarging and Wallet not exactly the same'
            self.log_message(message)
            
            if sum( np.isclose(self.AvailableMargin[:, self.tBinary], self.Wallet[:, self.tBinary], atol=2*self.tick) ) < self.Ntraders:
            
                breakpoint()
                idx = np.where( abs(self.AvailableMargin[:, self.tBinary]-self.Wallet[:, self.tBinary])>self.tick)[0]
                self.AvailableMargin[idx, self.tBinary] = self.Wallet[idx, self.tBinary]
            
            else:   
                
                self.AvailableMargin[:, self.tBinary] = self.Wallet[:, self.tBinary]             
                
        self.AvailableVolume[self.OrderBookAsks.ID.values, stock, self.tBinary] += self.OrderBookAsks.Volume
        self.OrderBookAs[stock], self.OrderBookBs[stock] = self.OrderBookAsks[0:0], self.OrderBookBids[0:0]
        
        #breakpoint() if sum(self.AvailableMargin[:, self.t] > self.Wallet[:, self.t]) else None
        if self.saveMicro:
            breakpoint() if sum(self.AvailableVolume[:, stock, self.t]) != sum(self.AvailableVolume[:, stock, self.tInit-1]) and self.phase != 'pretraining' else None
        else:
            breakpoint() if sum(self.AvailableVolume[:, stock, self.tBinary]) != self.InitialVolumeOwned*self.I and self.phase != 'pretraining' else None
               
    #######################################
    ############## Portfolio ##############               
    #######################################
    
    def wallet(self):
    
        breakpoint() if sum( np.isclose(self.AvailableMargin[:, self.tBinary], self.Wallet[:, self.tBinary], atol=self.tick) ) < self.Ntraders else None
        
        #Equity
        self.Aequity[:, :, self.tBinary] = self.VolumeOwned[:, :, self.tBinary]*np.reshape( np.repeat( self.Price[:, self.t], self.Ntraders ), [self.Ntraders, self.J])
        self.Atotal[:, self.tBinary] = self.Wallet[:, self.tBinary] + self.Aequity[:, :, self.tBinary].sum(axis=1)
            
    def withdrawing(self):
        
        # All these traders will sell (close) all their positions
        PeakToBottom = ( 1 - (self.Atotal[self.traders, :, self.tBinary] / self.Atotal[self.traders, :, 0] ) ) * 100
        
        self.losers = list(np.where(PeakToBottom > self.DrawdownLimit[self.traders])[0])
    
    def dividends(self):
        
        #Aequity over period of time or single time step?
        self.AvailableMargin[self.traders, self.tBinary] += self.Aequity[self.traders, :, self.tBinary].sum(axis=1)*self.D if np.mod(self.t+1, self.Ty) == 0 else 0
        
    def risk_free_interests(self):
        
        interests = (self.AvailableMargin[self.traders, self.tBinary])*self.RiskFreeRate
        interests = interests.round(2)
        
        self.AvailableMargin[self.traders, self.tBinary] += interests
        self.Wallet[self.traders, self.tBinary] += interests
        
        self.BalanceSheetBank[self.traders, self.tmicro] += interests
        
    #######################################
    ############## Time update ############             
    #######################################

    def new_time_step(self, t):
        
        self.Ntraders = len(self.traders)
                 
        self.t = t
        self.tmicro = t * self.saveMicro
        self.tBinary = t * self.saveMicro + (1-self.saveMicro)
        self.tForecast = t * self.saveMicro + self.maxTau * (1-self.saveMicro)

        self.tStar = ( self.tForecast-self.tau +1) #+ 1 * (1-self.saveMicro) #if tau is 1, tforecast-tau should be now
        self.tStarActionForecasting = self.t-self.tau

        variables = [
                     self.Wallet, #[:, self.tBinary-1:self.tBinary+1],
                     self.VolumeOwned, #[:, :, self.tBinary-1:self.tBinary+1],
                     self.AvailableVolume, #[:, :, self.tBinary-1:self.tBinary+1],
                     self.AvailableMargin, #[:, self.tBinary-1:self.tBinary+1],
                     self.Aequity, #[:, :, self.tBinary-1:self.tBinary+1],
                     self.Atotal, #[:, self.tBinary-1:self.tBinary+1],
                     self.averageBuyPrice, #[:, :, self.tBinary-1:self.tBinary+1]
                     ]

        #Portfolio
        if self.saveMicro:
            breakpoint() if self.VolumeOwned[:, :, t-1].sum() != self.VolumeOwned[:, :, self.tInit-1].sum() and self.phase not in ['pretraining'] else None
                
        for variable in variables:
            #t_ = 1 if self.saveMicro else 0
            #variable[..., t_] = variable[..., abs(t_-1)]

            if self.saveMicro:
            
                variable[..., t] = variable[..., t-1]
                
            elif not self.saveMicro:
            
                variable[..., :-1] = variable[..., 1:]
                variable[..., self.tBinary] = variable[..., self.tBinary-1]
        
        if not self.saveMicro:

            self.stateForecasting[..., :-1] = self.stateForecasting[..., 1:]
            #self.actionForecasting[..., :-1] = self.actionForecasting[..., 1:]
            self.Forecast[..., :-1, :] = self.Forecast[..., 1:, :]
            
            self.BalanceSheet = np.zeros((self.I, self.J, self.Tmicro, 2)) # trading expenses, trading gains and interest rate
            self.BalanceSheetBank = np.zeros((self.I, self.Tmicro)) # interest rate
            self.BalanceSheetFees = np.zeros((self.I, self.J, self.Tmicro))

            self.RewardForecasting = np.zeros((self.I, self.J, self.Tmicro, self.NactionsForecasting))
            self.RewardTrading = np.zeros((self.I, self.J, self.Tmicro, self.NactionsTrading))            
            
            self.statePricing = np.zeros((self.I, self.J, self.Tmicro, 2)).astype(int)
            self.actionPricing = np.zeros((self.I, self.J, self.Tmicro)).astype(int)
            self.RewardPricing = np.zeros((self.I, self.J, self.Tmicro, self.NactionsPricing))

        # Volatility
        #self.volatility = np.squeeze( np.array( [ self.returns[:, self.t-1-tau:self.t-1].var(axis=1) for tau in self.tauUnique ] ) )
        #self.volatilityTauIndexDict = dict( zip( self.tauUnique, [idx for idx, tau in enumerate( self.tauUnique )]) )
        
        #self.volatilityTw = self.returns[:, self.t-1-self.Tw:self.t-1].var(axis=1)
        #self.volatilityTw = np.maximum(self.volatilityTw, self.minimalVolalityTw) #replace 0 with smallest positive rational number 
        #self.volatilityLog = np.log10( self.volatilityTw )

        #self.volatilities = [self.volatilities[stock][1:] for stock in self.stocks] if len(self.volatilities[0]) >= self.Ty else self.volatilities
        #[self.volatilities[stock].append(self.volatilityLog[stock]) for stock in self.stocks]
        #self.volatilityReference[:] = [np.median(self.volatilities[stock]) for stock in self.stocks]
        
        #Fundamental Value Gap
        
        #self.fundamentalValueGap = np.zeros((self.I, self.J))
        
        #for stock in self.stocks:

        #    self.fundamentalValueGap[self.traders, stock] = np.median( abs( self.fundamentalValue[self.traders, stock, self.t-self.Ty:self.t] - self.Price[stock, self.t-self.Ty:self.t] ), axis=1)
        
        #Next price
        
        self.nextPriceForecast = np.zeros((self.I, self.J, self.NactionsForecasting))
        self.NextPriceReference = np.zeros((self.I, self.J))
        
        #RL
        if self.saveMicro:
        
            self.QForecasting[:, t] = self.QForecasting[:, t-1]
            self.VForecasting[:, t] = self.VForecasting[:, t-1]

            self.QTrading[:, t] = self.QTrading[:, t-1]
            self.VTrading[:, t] = self.VTrading[:, t-1]        
        
            self.QPricing[:, t] = self.QPricing[:, t-1]
            self.VPricing[:, t] = self.VPricing[:, t-1]
        
        #Market interaction
        self.TradingDecisionMakersBinary = self.TradingDecisionMakersBinaryMaster[:, self.t]
        self.NoTradingDecisionMakersBinary = (1-self.TradingDecisionMakersBinaryMaster[:, self.t]).astype(bool)
        self.TradingDecisionMakers = self.tradersArray[self.TradingDecisionMakersBinary]
        self.NoTradingDecisionMakers = self.tradersArray[self.NoTradingDecisionMakersBinary]
        
        # Orders
        self.OrderPrices = np.zeros((self.Ntraders, self.J)) #store order prices (bid, ask, limit, market, altogether; agents who pass will have 0)
        self.OrderVolumes = np.zeros((self.Ntraders, self.J)).astype(int)
        self.OrderCosts = np.zeros((self.Ntraders, self.J))
        self.OrderFees = np.zeros((self.Ntraders, self.J)) 

        self.DeltaWalletTrading = np.zeros(self.Ntraders)
        
        self.LastAuctionPriceReference = np.zeros((self.Ntraders, self.J))
        self.LimitPrices = np.zeros((self.Ntraders, self.J, self.NactionsPricingLimit))        
        self.PriceDeflection = np.zeros((self.Ntraders, self.J, self.NactionsPricingLimit))
        
    ##########################################
    ############### DECISIONS ################
    ##########################################
    
    #@njit    
    def decide_forecasting(self):
        '''
        Should all traders always learn forecasting despite non-interacting?
        '''

        self.forecast()
        
        for stock in self.stocks:
        
            s = self.state_forecasting(stock) #np.zeros((len(self.traders))).astype(int) #self.state_forecasting(stock)
            self.stateForecasting[self.traders, stock, self.tForecast] = s
            #self.actionForecasting[self.traders, stock, self.t] = self.softmax_decision_rule( self.QForecasting[self.traders, self.t, :2], self.traders )
            self.actionForecasting[self.traders, stock, self.t] = self.argmax_decision_rule( self.QForecasting[self.traders, self.tmicro, s, :self.NactionsForecasting], self.traders )
            #self.actionForecasting[self.traders, stock, self.t] = 0
            
            fundamentalists = (self.actionForecasting[self.traders, stock, self.t] == 1).astype(bool)
            self.NextPriceReference[self.traders, stock] = self.nextPriceForecast[self.traders, stock, 1] * fundamentalists + self.nextPriceForecast[self.traders, stock, 0] * (1-fundamentalists)
            self.NextPriceReference[self.traders, stock] = self.NextPriceReference[self.traders, stock].round(2)
            self.NextPriceReference[self.traders, stock] = np.maximum(self.NextPriceReference[self.traders, stock], self.tick) #price cannot be negative
            
    def state_trading(self):
    
        self.stateTrading[self.traders, :, self.tmicro, 0] = np.zeros((len(self.traders), self.J)) #np.array([ self.state_forecasting(stock) for stock in self.stocks]).T #np.zeros((len(self.traders), self.J)) #self.actionForecasting[self.traders, :, self.t]
        #self.stateTrading[self.traders, :, self.tmicro, 0] = np.array([ self.state_forecasting(stock) for stock in self.stocks]).T #np.zeros((len(self.traders), self.J)) #self.actionForecasting[self.traders, :, self.t]
        #self.stateTrading[self.traders, :, self.tmicro, 0] = self.actionForecasting[self.traders, :, self.tForecast]
        self.stateTrading[self.traders, :, self.tmicro, 1] = np.array([ self.state_BuyPrice(stock) for stock in self.stocks]).T #np.zeros((len(self.traders), self.J))
        
    def decide_trading(self):
            
        [self.action_Trading( self.TradingDecisionMakers, stock) for stock in self.stocks] if self.TradingDecisionMakersBinary.sum() > 0 else None  ### Decision Making on Trading ###
        
        if self.NoTradingDecisionMakersBinary.sum() > 0:
            self.actionTrading[ self.NoTradingDecisionMakers , :, self.tmicro] = 1

        self.actionTradingChosen[self.traders, :, self.tmicro] = self.actionTrading[self.traders, :, self.tmicro]
        
    def decide_pricing(self):

        gtilde = abs( np.random.normal( scale=self.g ) )
        self.gTildeFav = gtilde.copy()
        self.gTildeUnfav = gtilde /2

        #self.gTildeFav = abs( np.random.normal( scale=self.g ) )
        #self.gTildeUnfav = abs( np.random.normal( scale=self.g ) ) /2
        
        #self.gTildeFav = np.random.uniform(0, self.g, size=self.I)
        #self.gTildeUnfav = np.random.uniform(-self.g/2, 0, size=self.I)

        for stock in self.stocks:
                        
            self.trading_IDs_now(stock)

            self.statePricing[self.BidTraders, stock, self.tmicro, 0] = 0
            self.statePricing[self.AskTraders, stock, self.tmicro, 0] = 1

            s0 = self.statePricing[self.traders, stock, self.tmicro, 0] #bid or ask
            self.LastAuctionPriceReference[self.traders, stock] = self.priceDictionary[self.traders, stock, s0]
            
            self.sDummy = np.sign((s0==1)-.5) #sign for limit pricing
            
            self.price_deflection(self.TradingNow, stock)
            self.Price_limit_order(self.TradingNow, stock)

            self.statePricing[self.TradingNow, stock, self.tmicro, 1] = self.state_NextPrice(self.TradingNow, stock) #self.state_Volatility(stock)
            
            BidorAsk = self.statePricing[self.TradingNow, stock, self.tmicro, 0]
            s1 = self.statePricing[self.TradingNow, stock, self.tmicro, 1]

            if self.TradingNowCondition.sum() > 0:
                                                
                Q = self.QPricing[self.TradingNow, self.tmicro, BidorAsk, s1]                
                self.actionPricing[self.TradingNow, stock, self.tmicro] = self.softmax_decision_rule( Q, self.TradingNow )
                #self.actionPricing[self.TradingNow, stock, self.t] = self.argmax_decision_rule( Q, self.TradingNow )

                #self.volatilityFactor[self.TradingNow, stock] = self.volatility_factor(stock)

    def argmax_decision_rule(self, Q, traders):
    #def softmax_decision_rule(self, Q, traders):

        action = np.zeros(len(traders))
        action = np.argmax(Q, axis=1)

        Qmax = np.max(Q, axis=1)
        Qmax = Qmax*np.ones_like(Q).T
        Qmax = Qmax.T
        
        BooleanMax = (Q==Qmax)
        multipleMax = BooleanMax.sum(axis=1) > 1
        
        if multipleMax.sum() > 0:
        
            tradersPlainIndex = np.array(range(len(traders)))
            tradersMultipleMax = tradersPlainIndex[multipleMax]
            actionsMultipleMax = np.array(range(np.shape(Q)[1]))

            action[tradersMultipleMax] = [np.random.choice(actionsMultipleMax[BooleanMax[trader]]) for trader in tradersMultipleMax]
                
        return action
    
    def softmax_decision_rule(self, Q, traders):
        
        beta = self.beta[traders]*np.ones_like(Q.T)
        x = Q * beta.T
                
        x += x.mean(axis=1, keepdims=True)

        Nactions = np.shape(Q)[-1]
        x = np.maximum(ExpUnderFLowLimitMax/Nactions, x)
        x = np.minimum(ExpOverFLowLimitMax/Nactions, x)
        
        exp = np.exp(x)
        norm = exp.sum(axis=1, keepdims=True)
        #exp = exp.round(6)
        #norm = norm.round(6)
        #norm = np.maximum(norm, 10**(-6))
        
        probabilityActions = exp/norm
        #print( probabilityActions.sum(axis=1) )
        #probabilityActions = np.true_divide(exp, norm>0, 0)
                    
        rand = np.random.rand(len(traders))
        
        probabilityActionscumsum = probabilityActions.cumsum(axis=1)

        sample = ((rand*np.ones_like(probabilityActionscumsum.T)).T > probabilityActionscumsum).sum(axis=1) #add one when cum probability is crossed

        return sample

    def delta(self, traders, R, x, action): #, trade=False):
    
        delta = R - x #prediction error
        delta = delta.round(6) #avoid underflow
 
        alphaConf = ( self.alphaConf[traders]*np.ones_like(delta).T ).T
        alphaDisc = ( self.alphaDisc[traders]*np.ones_like(delta).T ).T
        
        #factual
        alphaFactual = action*( alphaConf * (delta > 0) + alphaDisc * (delta < 0) )
        
        #counterfactual
        alphaCounterfactual = (1-action)*( alphaConf * (delta < 0) + alphaDisc * (delta > 0) )
            
        alpha = alphaFactual + alphaCounterfactual

        breakpoint() if (alpha > 1).sum() > 0 else None
        
        return alpha * delta

    def delta_trading(self, traders, R, x):
    
        delta = R - x #prediction error
        delta = delta.round(6) #avoid underflow

        alpha = ( self.alphaV[traders]*np.ones_like(delta).T ).T
            
        return alpha * delta

    def delta_V(self, traders, R, x):
        
        delta = R - x #prediction error
        delta = delta.round(6) #avoid underflow

        alpha = self.alphaV[traders]
        
        return alpha * delta

    #################################################
    ############### FORECASTING #####################
    #################################################

    def state_forecasting_old(self, stock):
#    def state_forecasting(self, stock):

        CurrentGap = abs( self.fundamentalValue[self.traders, stock, self.t] - self.Price[stock, self.t-1] )
        gap = CurrentGap > self.fundamentalValueGap[self.traders, stock]
        state = gap.astype(int)
        
        return state

    def state_forecasting(self, stock):
#    def state_forecasting_new(self, stock):

        #DeltaT = np.sign( self.Forecast[self.traders, stock, self.tForecast, 0] - self.Price[stock, self.t-1] )
        #DeltaF = np.sign( self.Forecast[self.traders, stock, self.tForecast, 1] - self.Price[stock, self.t-1] )        
        #agreement = (DeltaT == DeltaF).astype(int)

        #CurrentGap = abs( self.fundamentalValue[self.traders, stock, self.t] - self.Price[stock, self.t-1] )
        #gap = CurrentGap > self.fundamentalValueGap[self.traders, stock]
        #gap = gap.astype(int)

        #GapChange = abs( self.fundamentalValue[self.traders, stock, self.t] - self.Price[stock, self.t-1] )
        #GapChangeOld = abs( self.fundamentalValue[self.traders, stock, self.t-1] - self.Price[stock, self.t-1] )
        
        GapChange = self.fundamentalValue[self.traders, stock, self.t] - self.Price[stock, self.t-1]
        #GapChangeOld = self.fundamentalValue[self.traders, stock, self.t-1] - self.Price[stock, self.t-1]
        
        fV = self.fundamentalValue[self.traders, stock, self.t-self.maxTau:self.t]
        fV = fV[self.TimeHorizonMask[self.traders]]
        
        PastPrice = self.Price[stock, self.t-self.maxTau:self.t]*self.TimeHorizonMask[self.traders]
        PastPrice = PastPrice[self.TimeHorizonMask[self.traders]]
        
        GapChangeOld = fV - PastPrice
        
        ratio = np.divide(GapChange, GapChangeOld, out=np.ones_like(GapChange), where=GapChangeOld!=0)
        
        OldGapTrue = (abs(GapChangeOld) > 0)
        OldGapFalse = 1-OldGapTrue
        
        biggerGap = ( (ratio > 1)*OldGapTrue + (abs(GapChange) > 0)*OldGapFalse ) #gap is bigger, same direction
        smallerGap = (ratio > 0)*(ratio <= 1)*OldGapTrue + (abs(GapChange) == 0)*OldGapFalse #gap is smaller or identical
        #revertedGap = 0*(ratio < 0)*OldGapTrue #gap reverted 
        
        gapChangeState = 2*biggerGap + smallerGap
        
        #state = 3*gap+gapChange     

        ##if abs(GapChangeOld) > 0:
        ##    gap = GapChange / GapChangeOld > 1 #gap is increasing in the same direction
        ##else:
        ##    gap = abs(GapChange) > 0 #gap is increasing
        
        ### Return autocorrelation ###
        #r = np.log(self.Price[stock, self.t-2*self.Tw:self.t])-np.log(self.Price[stock, self.t-1-2*self.Tw:self.t-1])
        
        #try:
        #    ac, _, pvalues = sm.tsa.acf(r, nlags=1, qstat=True) #[1]
        #    pvalues = pvalues[0]
        #    ac = ac[1]
        #except: #when all values are the same and no acf can be computed
        #    print('**Except**')
        #    ac = 0
        #    pvalues = 1
        
        #acSignTrue = (pvalues < 0.05)
        
        #acPositive = (ac>0)*acSignTrue
        #acNone = 1-acSignTrue
        
        #acState = (2*acPositive + acNone) #+ 0*acSignTrue*(ac<0)
        #print('AC negative') if acState == 0 else print('AC null') if acState == 1 else print('AC positive') if acState == 2 else None
        #acState *= np.ones_like(gap)

        #state = gap+3*acState

        ### Returns autocorrelation
        
        #ac, _, pvalues = sm.tsa.acf(self.Price[stock, self.t-self.Tw:self.t], nlags=1, qstat=True) #[1]

        #s = np.log(self.Price[stock, self.t-2*self.Tw:self.t])-np.log(self.Price[stock, self.t-2*self.Tw-1:self.t-1])
        #acTemp, _, pvaluesTemp = sm.tsa.acf(s, nlags=1, qstat=True)  
        #ac = acTemp[1]*np.ones(len(gap))
        #pvalues = pvaluesTemp[0]*np.ones_like(ac)
        
        #ac = np.zeros(len(gap))
        #pvalues = np.zeros_like(ac)

        #for taun, tau in enumerate(self.tauUnique):
        
            #traders = self.tradersArray[self.tau==tau]
            ###s = self.Price[stock, self.t-tau:self.t]
            #s = np.log(self.Price[stock, self.t-tau:self.t])-np.log(self.Price[stock, self.t-tau-1:self.t-1])
            #acTemp, _, pvaluesTemp = sm.tsa.acf(s, nlags=1, qstat=True)  
            ###print(acTemp[1], pvaluesTemp[0]) if acTemp[1]<0 and pvaluesTemp[0] <0.05 else None
            ###print(acTemp[1], pvaluesTemp[0], tau) if acTemp[1]>0 and pvaluesTemp[0] <0.05 and tau< 25 else None
            #ac[traders] = acTemp[1]
            #pvalues[traders] = pvaluesTemp[0]
            
        #acSignTrue = (pvalues < 0.05)
        
        #acPositive = (ac>0)*acSignTrue
        #acNone = 1-acSignTrue
        
        #acState = (2*acPositive + acNone) #+ 0*acSignTrue*(ac<0)
        ##print(set(acState))
        ##print(acSignTrue.mean(), acState.mean())

        #state = gapChange+3*acState

        ### Trend from pvalues of slope in forecast_trend ###
        #breakpoint()
        trend = self.pvaluesSlope.copy()
        
        #traders = self.tradersArray[self.tau<=2]
        #print(set(trend[traders]))
        
        #print(np.mean(trend))        
        state = (gapChangeState+3*trend).astype(int)
        
        #print(set(state))
        
        return state #gap.astype(int)

    def compute_p_value(self, x, y, slope):
    
        n = len(x)
    
        # Residuals: y - slope * x
        residuals = y - slope * x

        # Residual Sum of Squares (RSS)
        rss = np.sum(residuals**2)

        # Estimate of variance (sigma^2)
        sigma_squared = rss / (n - 1)

        # Standard error of the slope
        x_variance = np.sum(x**2)
        standard_error = np.sqrt(sigma_squared / x_variance)

        # Compute t-statistic for the slope
        if standard_error == 0:  # Edge case where standard error is zero
            t_statistic = np.inf if slope != 0 else 0
        else:
            # Compute t-statistic for the slope
            t_statistic = slope / standard_error

        # Compute the two-tailed p-value from the t-statistic
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n-1))

        return p_value


    def forecast_trend(self, stock):
        
        for taun, tau in enumerate(self.tauUnique):

            traders = self.tradersArray[self.tau==tau]

            ##m, b = np.polyfit(x[:-1], y, 1)

            ##pvalues for significant slope (intercept is not forced to zero, though!)
            ##res = linregress(x[:-1], y)
            
            #x = np.arange(0, tau)
            #y = self.Price[stock, self.t-tau:self.t]
            #res = linregress(x, y)
            #self.pvaluesSlope[traders] = res.pvalue<0.01

            x = np.arange(-tau, 1) #to center at origin (0) 
            y = self.Price[stock, self.t-tau:self.t] - self.Price[stock, self.t-1] #subtract to center at origin (0)
            
            #Faster solution
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            m = np.linalg.lstsq(x[:-1].reshape(-1,1), y, rcond=-1)[0][0]
                        
            trend = m*(-x[::-1])

            # Calculate p-value for the slope
            p_value = self.compute_p_value(x[:-1], y, m)
            
            # Store the p-value comparison result
            self.pvaluesSlope[traders] = p_value < 0.01
            #print(tau, set(self.pvaluesSlope[traders])) if tau < 10 else None
            
            #x = x.reshape(-1, 1)
            #y = y.reshape(-1, 1)
            #trend2 = LinearRegression(fit_intercept=False).fit(x[:-1], y).predict( -x[::-1] ) #x[-1] to take only tau points (including 0)
            #breakpoint() if abs(trend[-1]-trend2[-1]) > 0 else None

            self.nextPriceForecast[traders, stock, 0] = self.Price[stock, self.t-1] + trend[1]
            self.Forecast[traders, stock, self.tForecast, 0] = self.Price[stock, self.t-1] + trend[-1] #trend-following
            
            if abs(trend[-1]) < self.tick:

                message = 'no trend'
                self.log_message(message)
            
                trendCorrection = np.sign(np.random.rand( len(traders) ) -.5)*self.tick #flip coin for tick trend when no trend
                self.Forecast[traders, stock, self.tForecast, 0] += trendCorrection
                
    def forecast(self):

        [self.forecast_trend(stock) for stock in self.stocks] #forecast first tool (chartist: trend-following)
        self.Forecast[:, :, self.tForecast, self.NactionsForecasting-1] = self.fundamentalValue[:, :, self.t] #fundamentalist
        
        if self.NactionsForecasting > 2:
            self.Forecast[:, :, self.t, 1] = (self.Forecast[:, :, self.t, 0]+self.Forecast[:, :, self.t, -1])/2 #mixed #AVERAGED BEFORE MIN TICK TRANSFORMATION! Here TF can be negative, but then stored as tick (positive)
        
                
        for stock in self.stocks:
        
            #Fundamentalism
            self.nextPriceForecast[self.traders, stock, self.NactionsForecasting-1] = self.Price[stock, self.t-1]
            
            #fundamentalDrive = np.minimum(10 * self.kappa / self.tau, 1)
            #fundamentalDrive = np.minimum( (1 + self.kappa) / self.tau, 1)
            #fundamentalDrive = np.minimum( (self.kappa) / self.tau, 1)
            #fundamentalDrive = self.kappa / self.tau
            fundamentalDrive = 1 / self.tau
            
            self.nextPriceForecast[self.traders, stock, self.NactionsForecasting-1] += fundamentalDrive * (self.fundamentalValue[self.traders, stock, self.t] - self.Price[stock, self.t-1]) #(1+self.kappa)
            
            #Mixed
            if self.NactionsForecasting > 2:
            
                self.nextPriceForecast[self.traders, stock, 1] = (self.nextPriceForecast[self.traders, stock, 0]+self.nextPriceForecast[self.traders, stock, -1])/2 
            
            self.nextPriceForecast[self.traders, stock] = self.nextPriceForecast[self.traders, stock].round(2)
        
            self.Forecast[self.traders, stock, self.tForecast] = np.maximum( self.Forecast[self.traders, stock, self.tForecast], self.tick )
        
        self.Forecast[self.traders, :, self.tForecast] = self.Forecast[self.traders, :, self.tForecast].round(2)
        
        #self.plot_forecasts(stock) #Diagnostics
        
    def plot_forecasts(self, stock):  
            
        if self.t > self.tInitLearn and np.mod(self.t, 250) == 0:

            import matplotlib.cm as cm
            plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial']})
            plt.rcParams["axes.spines.right"] = False
            plt.rcParams["axes.spines.top"] = False
            import string
            
            tradersPlot = np.random.choice(self.I, size=100)
            fig, axs = plt.subplots(figsize=(9, 6), nrows=2, ncols= 2, sharey=True)
            
            c = self.kappa
            col = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
            
            axs[0, 0].plot(range(self.t-self.maxTau, self.t), self.Price[0, self.t-self.maxTau:self.t], color='k', label=r'$P$')
            [axs[0, 0].plot(range(self.t, self.t+self.tau[trader]), np.linspace(self.Price[0, self.t-1], self.Forecast[trader, 0, self.tForecast, 1], self.tau[trader]), c=col[trader] ) for trader in tradersPlot]
            axs[0, 0].plot(range(self.t-self.maxTau, self.t), self.fundamentalValueTrue[0, self.t-self.maxTau:self.t], color=[.5, .5, .5], label=r'$f$')
            axs[0, 0].plot(range(self.t, self.t+self.maxTau), self.fundamentalValueTrue[0, self.t]*np.ones(self.maxTau), color=[.5, .5, .5], ls='--')
            
            #axs[0, 0].plot(range(self.t-self.maxTau, self.t), self.fundamentalValue[:, 0, self.t-self.maxTau:self.t].mean(axis=0), color=[.8, .8, .8], label=r'$\mu_{\overset{\sim}{f}}$')
            axs[0, 0].plot(range(self.t-self.maxTau, self.t), np.median(self.fundamentalValue[:, 0, self.t-self.maxTau:self.t], axis=0), color=[.8, .8, .8], label=r'$M_{\overset{\sim}{f}}$')
            axs[0, 0].set_xlabel(r'$t$')
            axs[0, 0].set_ylabel('Price')

            axs[0, 0].legend()

            c = self.tau
            col = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
            
            axs[0, 1].plot(range(self.t-self.maxTau, self.t), self.Price[0, self.t-self.maxTau:self.t], color='k')
            [axs[0, 1].plot(range(self.t, self.t+self.tau[trader]), np.linspace(self.Price[0, self.t-1], self.Forecast[trader, 0, self.tForecast, 0], self.tau[trader]), c=col[trader] ) for trader in tradersPlot]
            axs[0, 1].set_xlabel(r'$t$')

            c = self.kappa
            col = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
            
            axs[1, 0].axhline(y=self.Price[0, self.t-1], color='k', label=r'$P_{t-1}$')
            axs[1, 0].scatter(self.kappa, self.Forecast[:, 0, self.tForecast, 1], s=1, c=col)
            axs[1, 0].set_xlabel(r'$\kappa$')
            axs[1, 0].set_ylabel('Price')
            
            axs[1, 0].axhline(y=self.fundamentalValueTrue[0, self.t], color=[.5, .5, .5], label=r'$f_t$')
            #axs[1, 0].axhline(y=self.fundamentalValue[:, 0, self.t].mean(axis=0), color=[.8, .8, .8], label=r'$\mu_{\overset{\sim}{f}, t}$')
            axs[1, 0].axhline(y=np.median(self.fundamentalValue[0, :, self.t], axis=0), color=[.8, .8, .8], label=r'$M_{\overset{\sim}{f}}$')
            
            axs[1, 0].legend(loc='lower right')
            
            c = self.tau
            col = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
            
            axs[1, 1].axhline(y=self.Price[0, self.t-1], color='k')            
            axs[1, 1].scatter(self.tau, self.Forecast[:, 0, self.tForecast, 0], s=1, c=col)
            axs[1, 1].set_xlabel(r'$\tau$')
            
            
            axs = axs.flatten()
            
            for axn, ax in enumerate(axs):
            
                ax.text(-0.1, 1.0, string.ascii_lowercase[axn], transform=ax.transAxes, size=12, weight='bold')
                        
            plt.tight_layout()
            plt.savefig(f'Forecasts/{self.t}_2.pdf', dpi=300)
            #plt.show()
            plt.close()
        
    #############################################
    ################## TRADING ##################
    #############################################

    def state_BuyPrice(self, stock):

        actionMask = np.zeros((len(self.traders), self.NactionsForecasting)).astype(bool)
        actionMask[ self.traders, self.actionForecasting[self.traders, stock, self.t] ] = True

        Forecast = self.Forecast[self.traders, stock, self.tForecast]
    
        H = np.squeeze( Forecast[actionMask] )
        #P = np.squeeze( self.Price[stock, self.t-1] )
        
        P = self.NextPriceReference[self.traders, stock] #*(1-self.b)
        
        Pavg = np.squeeze( self.averageBuyPrice[:, stock, self.tBinary] )

        PH = P < H
        PavgH = Pavg < H
        PavgP = Pavg < P

        VolumeCondition = self.VolumeOwned[:, stock, self.tBinary] > 0
        
        state = (1-VolumeCondition)*(1* (1-PH) + 5* PH )
        state += VolumeCondition * (1-PH) * ( 2*PavgH + 3*(1-PavgH)*PavgP + 4*(1-PavgP) ) #*2 (one state only, for selling conditions H<P)
        state += VolumeCondition * PH * ( 6*PavgP + 7*PavgH*(1-PavgP) + 8*(1-PavgH) )
        
        #breakpoint() if 0 in state else None
        
        state -= 1
        
        return state
        
    #def state_Volatility(self, stock):

    #    volatilityLevel = int(self.volatilityLog[stock] > self.volatilityReference[stock])

    #    return volatilityLevel 

    def action_Trading(self, traders, stock):
            
        ''' a0 = 0 corresponds to a buy order of a floored quantity Ai bonds(t)/[P_bid(t)J] of stocks j for a bid price of self.Price i,j
            a0 = 1 corresponds to no order being sent to the order book (the agent simply holds its position), and
            a0 = 2 corresponds to a sell order of a quantity Qi,j (t) of stocks j for an ask price of P_ask(t)
        bid(t).'''
        
        VolumeCondition = self.VolumeOwned[traders, stock, self.tBinary] > 0

        if VolumeCondition.sum() > 0:
            volumeOwnersIndex = np.nonzero(VolumeCondition) #[0]
            volumeOwners = traders[ volumeOwnersIndex ]
            
            s0 = self.stateTrading[volumeOwners, stock, self.tmicro, 0]
            s1 = self.stateTrading[volumeOwners, stock, self.tmicro, 1]
            
            self.actionTrading[volumeOwners, stock, self.tmicro] = self.softmax_decision_rule( self.QTrading[volumeOwners, self.tmicro, s0, s1], volumeOwners ) #
            #self.actionTrading[volumeOwners, stock, self.t] = self.argmax_decision_rule( self.QTrading[volumeOwners, self.tmicro, s0, s1], volumeOwners ) #
            
        VolumeCondition = 1-VolumeCondition

        if VolumeCondition.sum() > 0:

            volumeFreeIndex = np.nonzero(VolumeCondition)
            volumeFree = traders[volumeFreeIndex]
            
            s0 = self.stateTrading[volumeFree, stock, self.tmicro, 0]
            s1 = self.stateTrading[volumeFree, stock, self.tmicro, 1]
            
            #no selling available
            self.actionTrading[volumeFree, stock, self.tmicro] = self.softmax_decision_rule( self.QTrading[volumeFree, self.tmicro, s0, s1, :2], volumeFree )
            #self.actionTrading[volumeFree, stock, self.t] = self.argmax_decision_rule( self.QTrading[volumeFree, self.tmicro, s0, s1, :2], volumeFree )
        
    #############################################
    ################## PRICING ##################
    #############################################

    def state_NextPrice(self, traders, stock):

        s1 = self.NextPriceReference[traders, stock] < self.LastAuctionPriceReference[traders, stock]

        return s1 

    #def volatility_factor(self, stock):
    
    #    Delta = self.volatilityLog[stock] - self.volatilityReference[stock]
        #print(Delta)
    
    #    return (1 + Delta) * (Delta >= 0) + (1/(1+abs(Delta))* (Delta < 0))#1
    
   #############################################
   ################## Returns ##################
   #############################################

    def observe_outcomes(self, stock):

        self.reward_Forecasting(stock)
        self.reward_Pricing(stock)

    # Define the sigmoid and its derivative
    def sigmoid(self, x, nu):
        
        sigma = 1 / (1 + np.exp(-nu * x ))
        
        #try:
        #    sigma = 1 / (1 + np.exp(-nu * x ))
        #except OverflowError:
        #    sigma = 1
        #except: # UnderflowError:
        #    sigma = 0
            
        return sigma

    def sigmoid_prime(self, x, nu):
        sigma = self.sigmoid(x, nu)
        return sigma * (1 - sigma)

    # Define the reward functions
    def reward_Forecasting_overshoot(self, x):
        return self.sigmoid_prime(x, self.nu_plus) / self.sigmoid_prime_plus_norm * .5 + .5
    
    def reward_Forecasting_undershoot(self, x):
        return 2 * self.sigmoid_prime(x, self.nu_minus) / self.sigmoid_prime_minus_norm - 1

    def reward_Forecasting_zero(self, x):
        return 2 * self.sigmoid_prime(x, self.nu_zero) / self.sigmoid_prime_zero_norm - 1

    def reward_Forecasting(self, stock):
        
        Forecast = self.Forecast[self.traders, stock, self.tForecast-self.maxTau+1:self.tForecast+1]
        Forecast = Forecast[self.TimeHorizonMask[self.traders]]
                
        PastPrice = self.Price[stock, self.t-self.maxTau+1:self.t+1]*self.TimeHorizonMask[self.traders]
        PastPrice = PastPrice[self.TimeHorizonMask[self.traders]]

        for actionF in range(self.NactionsForecasting):

            ### R^{\Delta} #agents expecting price change

            idx = np.nonzero(Forecast[:, actionF]-PastPrice)[0]

            x = (self.Price[stock, self.t] - PastPrice[idx])/ (Forecast[idx, actionF]-PastPrice[idx])
            x = np.squeeze(x)
            
            #To avoid under and overflox in sigmoid exponential
            x = np.minimum(x, 10)
            x = np.maximum(x, -10)
                        
            #x = np.linspace(-3, 3, len(x))
                        
            idxUnder = x<1
            idxOver = x>=1 #(1-idxUnder).astype(bool)
            
            self.RewardForecasting[idx[idxUnder], stock, self.tmicro, actionF] = self.reward_Forecasting_undershoot(x[idxUnder]-1) 
            self.RewardForecasting[idx[idxOver], stock, self.tmicro, actionF] = self.reward_Forecasting_overshoot(x[idxOver]-1)
            
            #Diagnostics
            #plt.scatter(x, self.RewardForecasting[idx, stock, self.tmicro, actionF]), plt.xlim([-3, 3]), plt.show()

            ### R^{0} #agents not expecting price change
            
            idxzero = np.nonzero(Forecast[:, actionF] == PastPrice) #R_0            
            x = abs(self.Price[stock, self.t]/PastPrice[idxzero] -1) 

            #To avoid under and overflox in sigmoid exponential
            x = np.minimum(x, 10)
            x = np.maximum(x, -10)

            self.RewardForecasting[idxzero, stock, self.tmicro, actionF] = self.reward_Forecasting_zero(x) 
                                                        
    def impossible_counterfactual(self, stock, tradersBinary, unchosenAction):
        
        #Impossible counterfactual -> no learning, R = Q
        
        if tradersBinary.sum() > 0:
        
            traders = np.nonzero(tradersBinary)[0]
            traders = self.tradersArray[tradersBinary]
            
            s0 = self.stateTrading[traders, stock, self.tmicro, 0]
            s1 = self.stateTrading[traders, stock, self.tmicro, 1]
                            
            self.RewardTrading[traders, stock, self.tmicro, unchosenAction] = self.QTrading[traders, self.tmicro, s0, s1, unchosenAction] #no surprise, no learning
    
    #***model-based learning of trading*** before trading (with Price[t-1]) would improve decision and forecast-posteriori problem,
            # but how could traders learn from environment?

    def reward_Trading(self, stock):
        
        #Bid counterfactual conditions
        #Price = max(self.Price[stock, self.t-1], self.Price[stock, self.t] )*(1+self.b)
        
        self.PriceReferenceRewardTrading = self.NextPriceReference[self.traders, stock]
        
        self.cash_to_invest()
        
        self.BidOrderVolumes = (self.CashToInvest[self.traders]/ (self.PriceReferenceRewardTrading*(1+2*self.b)) ).astype(int)
        self.BidOrderFees = self.BidOrderVolumes * self.PriceReferenceRewardTrading * self.b
        self.BidOrderTotalTradingCosts = self.BidOrderVolumes*self.PriceReferenceRewardTrading*(1+self.b)
        
        #Ask counterfactual conditions   
        self.AskOrderVolumes = self.Volume_ask(self.traders, stock) #for buyers, available volume at time t (now) is different from value at beginning of time t, as agent bought volume (also depending on auction outcome), so consider t-1 (end)
        self.AskOrderFees = self.AskOrderVolumes * self.PriceReferenceRewardTrading * self.b
        
        AskPossible = ( self.AskOrderVolumes > 0)*( self.Wallet[self.traders, self.tBinary] > self.AskOrderFees )
        BidPossible = ( self.BidOrderVolumes > 0)*( self.Wallet[self.traders, self.tBinary] > self.BidOrderTotalTradingCosts )

        AskImpossible = (1-AskPossible).astype(bool)
        BidImpossible = (1-BidPossible).astype(bool)
        
        ### Indeces ###

        VolumeTradersBinary = (self.VolumeOwned[self.traders, stock, self.tBinary]>0)
        noVolumeTradersBinary = (1-VolumeTradersBinary).astype(bool) #(self.VolumeOwned[self.traders, stock, self.tBinary]==0)
        
        noVolumeTraders = np.nonzero(noVolumeTradersBinary)[0]
        noVolumeTraders = self.tradersArray[noVolumeTraders]

        VolumeTraders = np.nonzero(VolumeTradersBinary)[0]
        VolumeTraders = self.tradersArray[VolumeTraders]
        
        ### Factuals ###

        self.RewardTrading[noVolumeTraders, stock, self.tmicro, 0] = self.reward_openers(stock, noVolumeTraders)        
        self.RewardTrading[noVolumeTraders, stock, self.tmicro, 1] = self.reward_waiters(stock, noVolumeTraders)
        self.RewardTrading[VolumeTraders, stock, self.tmicro, 0] = self.reward_buyers(stock, VolumeTraders)        
        self.RewardTrading[VolumeTraders, stock, self.tmicro, 1] = self.reward_holders(stock, VolumeTraders)
        self.RewardTrading[VolumeTraders, stock, self.tmicro, 2] = self.reward_sellers(stock, VolumeTraders)
         
        #*** back to normal forecast chartist for more movement!  **** or why price movement difference between micro and no micro?
        
        #breakpoint()
                   
        ### Wrap-up reward ###
        
        ones = np.ones_like(self.RewardTrading[self.traders, stock, self.tmicro]).T
        
        Atotal = ( self.Atotal[self.traders, self.tBinary-1]*ones ).T
        norm = self.rhoBid[self.traders] #* np.sqrt( self.tau[self.traders] )
        norm = ( norm*ones ).T
        self.RewardTrading[self.traders, stock, self.tmicro] /= norm #self.tau[self.traders]/self.rhoBid[self.traders]
                
        W = np.maximum(self.tick, Atotal+self.RewardTrading[self.traders, stock, self.tmicro] )
                
        self.RewardTrading[self.traders, stock, self.tmicro] = np.log( W ) - np.log( Atotal ) #log( (W+r)/W )
        
        #self.RewardTrading[self.traders, stock, self.tmicro] = np.tanh( self.RewardTrading[self.traders, stock, self.tmicro] )

        self.RewardTrading[self.traders, stock, self.tmicro] *= self.rewardMultiplier
        
        #self.RewardTrading[self.traders, stock, self.t] /= (self.beta*np.ones_like(self.RewardTrading[self.traders, stock, self.t]).T ).T
        
        #R = Q
        ### Impossible action ###                
        tradersBinary = noVolumeTradersBinary*BidImpossible
        self.impossible_counterfactual(stock, tradersBinary, unchosenAction=0)

        tradersBinary = VolumeTradersBinary*BidImpossible
        self.impossible_counterfactual(stock, tradersBinary, unchosenAction=0)

        tradersBinary = VolumeTradersBinary*AskImpossible
        self.impossible_counterfactual(stock, tradersBinary, unchosenAction=2)
        
        #tradersi = (self.stateTrading[:, 0, self.tmicro, 1] == 1)
        #tradersi2 = self.RewardTrading[tradersi, stock, self.tmicro, 1] > self.RewardTrading[tradersi, stock, self.tmicro, 2]
        #tradersi3 = ( self.Forecast[tradersi, stock, self.tForecast, self.actionForecasting[tradersi, stock, self.tForecast]] > self.PriceReferenceRewardTrading[tradersi] )
        #print(sum(tradersi), sum(tradersi3), sum(tradersi2), '\n')
        
        #RewardCheck = abs(self.RewardTradingFactual[self.traders, stock, self.t]) > 1000
        #print(np.where(RewardCheck), self.RewardTradingFactual[np.where(RewardCheck), stock, self.t])  if (RewardCheck).sum() > 0 else None        

    def reward_waiters(self, stock, traders):
    
        DeltaRiskFree = self.AvailableMargin[traders, self.tBinary]*self.RiskFreeRate
      
        reward = DeltaRiskFree

        return reward
    
    def reward_openers(self, stock, traders):
       
        Forecast = self.Forecast[traders, stock, self.tForecast, self.actionForecasting[traders, stock, self.t]]

        VolumeOwned = self.BidOrderVolumes[traders] #self.InitialVolumeOwned+
        
        Delta = (Forecast - self.PriceReferenceRewardTrading[traders])*VolumeOwned
        
        DeltaFees = self.BidOrderFees[traders] #may have not bought the whole order
        DeltaOrder = self.BidOrderTotalTradingCosts[traders]
        DeltaRiskFree = (self.Wallet[traders, self.tBinary]-DeltaOrder)*self.RiskFreeRate

        reward = Delta - DeltaFees + DeltaRiskFree #Delta order? apply +gamma*Q_max?
        
        return reward
        
    def reward_buyers(self, stock, traders):
    
        Forecast = self.Forecast[traders, stock, self.tForecast, self.actionForecasting[traders, stock, self.t]]

        VolumeOwned = self.VolumeOwned[traders, stock, self.tBinary]+self.BidOrderVolumes[traders]
        
        averageBuyPrice = self.averageBuyPrice[traders, stock, self.tBinary]*self.VolumeOwned[traders, stock, self.tBinary]
        averageBuyPrice += self.PriceReferenceRewardTrading[traders] * self.BidOrderVolumes[traders]
        
        averageBuyPrice /= VolumeOwned #multiplying again in next line though; clearer this way, but computationally not efficient
                
        Delta = (Forecast - averageBuyPrice)*VolumeOwned #(Forecast - self.PriceReferenceRewardTrading[traders])*VolumeOwned
        
        DeltaAvgPrice = 0 #(self.averageBuyPrice[traders, stock, self.tBinary-1] - averageBuyPrice)*VolumeOwned
        
        DeltaFees = self.BidOrderFees[traders] #may have not bought the whole order
        DeltaOrder = self.BidOrderTotalTradingCosts[traders]
        DeltaRiskFree = (self.Wallet[traders, self.tBinary]-DeltaOrder)*self.RiskFreeRate
                
        reward = Delta - DeltaFees + DeltaRiskFree #DeltaAvgPrice*(Delta>0)
        
        return reward 

    def reward_holders(self, stock, traders):

        Forecast = self.Forecast[traders, stock, self.tForecast, self.actionForecasting[traders, stock, self.t]]
        
        VolumeOwned = self.VolumeOwned[traders, stock, self.tBinary]
        averageBuyPrice = self.averageBuyPrice[traders, stock, self.tBinary]
        
        #Delta = (Forecast - self.PriceReferenceRewardTrading[traders])*self.VolumeOwned[traders, stock, self.tBinary] #did not change for holders; buyers increased it at t; sellers sold it; t-1 reflects right quantity in all cases
        Delta = (Forecast - averageBuyPrice)*VolumeOwned
        
        DeltaRiskFree = self.AvailableMargin[traders, self.tBinary]*self.RiskFreeRate
        
        reward = Delta + DeltaRiskFree

        return reward
    
    def reward_sellers(self, stock, traders):

        Forecast = self.Forecast[traders, stock, self.tForecast, self.actionForecasting[traders, stock, self.t]]

        averageBuyPrice = self.averageBuyPrice[traders, stock, self.tBinary]
        
        Delta = (self.PriceReferenceRewardTrading[traders] - averageBuyPrice)*self.AskOrderVolumes[traders]

        DeltaFees = self.AskOrderFees[traders]
        DeltaRiskFree = (self.Wallet[traders, self.tBinary]-DeltaFees)*self.RiskFreeRate
                
        reward = Delta - DeltaFees + DeltaRiskFree
        
        return reward 
              
    def reward_Pricing(self, stock):
        
        self.trading_IDs(stock)
                
        for actionPricing in range(self.NactionsPricingLimit): #exclude market orders
            
            #OrderPrices = self.LimitPrices[self.AskLimitTraders, stock, actionPricing]
            #self.RewardPricing[self.AskLimitTraders, stock, self.tmicro, actionPricing] = self.reward_limit( OrderPrices, stock, self.AskLimitTraders, ask=True)
            
            OrderPrices = self.LimitPrices[self.AskTraders, stock, actionPricing]
            self.RewardPricing[self.AskTraders, stock, self.tmicro, actionPricing] = self.reward_limit( OrderPrices, stock, self.AskTraders, ask=True)
            
            #OrderPrices = self.LimitPrices[self.BidLimitTraders, stock, actionPricing]                
            #self.RewardPricing[self.BidLimitTraders, stock, self.tmicro, actionPricing] = self.reward_limit( OrderPrices, stock, self.BidLimitTraders, ask=False)
            
            OrderPrices = self.LimitPrices[self.BidTraders, stock, actionPricing]                
            self.RewardPricing[self.BidTraders, stock, self.tmicro, actionPricing] = self.reward_limit( OrderPrices, stock, self.BidTraders, ask=False)

        #MarketOrderReward = self.MarketCondition[self.TradingNow]*0
        
        self.RewardPricing[self.TradingNow, stock, self.tmicro, -1] = 0 #-0.5 #0 #-.5
        
        #self.RewardPricing[self.LimitTraders, stock, self.tmicro] *= 10
        self.RewardPricing[self.TradingNow, stock, self.tmicro] *= 5 #10
      
        #Diagnostic by plotting

        #buyers = (self.actionTrading[self.traders, stock, self.t] == 0)       # chosen, not effective  
        #buyers = np.where(buyers)

        #sellers = (self.actionTrading[self.traders, stock, self.t] == 2)        # chosen, not effective
        #sellers = np.where(sellers)

        #fig, axs = plt.subplots(5, 3, sharex=True, sharey=True)
        #axs = axs.flatten()
        #for traders, traders_name, color in zip([buyers, sellers], ['buyers', 'sellers'], ['g', 'r']):
            
            #for trader in traders[0]:
                
                #axs[self.actionPricing[trader, stock, self.t]].scatter(trader, self.RewardPricingFactual[trader, stock, self.t], label=traders_name, color=color, alpha=.2)
                #axs[self.actionPricing[trader, stock, self.t]].axhline(y=0, color='k', alpha=.3)
        ##plt.legend()
        #plt.tight_layout()
        ##plt.title(self.Price[stock, self.t]-self.Price[stock, self.tmicro-1])
        #plt.show() #plt.ylim([-.25, .25]), plt.show()
                      
    def reward_limit(self, OrderPrices, stock, traders, ask):
        
        unfavPricing = (self.actionPricing[traders, stock, self.tmicro] % 2).astype(bool) #favourite (0) and unfavourite (1) price deflection direction
        favPricing = (1-unfavPricing)
        gAdj = self.g[traders] * (2*favPricing + unfavPricing / 2) #g adjusted to price deflection preference #eta50 in thesis
        
        etaResolved = self.etaResolved / gAdj
        etaFail = self.etaFail / gAdj
        
        #etaResolved = self.etaResolved / 0.01
        #etaFail = self.etaFail / 0.01
        
        referencePrice = self.Price[stock, self.t]
    
        if self.Volume[stock, self.t] < 1: #Differentiating based on intersection or not
            referencePrice = self.priceDictionary[traders, stock, 0] if ask else self.priceDictionary[traders, stock, 1] if not ask else None #Differentiating based on intersection or not
        
        x = (OrderPrices/referencePrice) - 1
        
        if ask:
            PriceCondition = x <=0 #may have been good price, but it could be unresolved because of too many market orders on one side
            y = np.maximum(etaResolved*x, ExpUnderFLowLimitMax)
        elif not ask:
            PriceCondition = x >=0 #
            y = np.maximum(-etaResolved*x, ExpUnderFLowLimitMax)        
        
        y = np.minimum(y, ExpOverFLowLimitMax)
        
        ResolvedOrderReward = PriceCondition*np.exp( y )
        #ResolvedOrderReward = PriceCondition*( 1.5 * np.exp( y ) - 0.5)

        if ask:
            PriceCondition = (x >0) 
            y = np.maximum(-etaFail*x, ExpUnderFLowLimitMax)
        elif not ask:
            PriceCondition = (x <0)
            y = np.maximum(etaFail*x, ExpUnderFLowLimitMax)        
        
        y = np.minimum(y, ExpOverFLowLimitMax)        
        
        FailedOrderReward = PriceCondition*( np.exp( y )/2 -1 ) 
        
        return ResolvedOrderReward+FailedOrderReward

    ######################################
    ############ LEARNING ################
    ######################################

    def learn_forecasting(self, stock):

        self.actionForecastingBinary = np.zeros(( len(self.traders), self.NactionsForecasting )).astype(bool)
        self.actionForecastingBinary[ self.traders, self.actionForecasting[self.traders, stock, self.tStarActionForecasting] ] = True

        R = self.RewardForecasting[self.traders, stock, self.tmicro]
        self.Q_Forecasting_update(R, stock)            

    def learn_trading(self, stock):
    
        self.actionTradingBinary = np.zeros(( len(self.traders), self.NactionsTrading )).astype(bool)
        self.actionTradingBinary[ :, self.actionTrading[self.traders, stock, self.tmicro] ] = True
        
        R = self.RewardTrading[self.traders, stock, self.tmicro] #+ 0.5 * np.max(self.QTrading[self.t, s0, s1, s2]) #add also in counterfactual counterfactual_Trading, in case
        self.Q_Trading_update(self.traders, stock, R) #self.actionTradingEffective
                
    def learn_pricing(self, stock):
        
        self.actionPricingBinary = np.zeros(( len(self.TradingNow), self.NactionsPricing)).astype(bool)
        
        self.actionPricingBinary[ :, self.actionPricing[self.TradingNow, stock, self.tmicro] ] = True
            
        R = self.RewardPricing[self.TradingNow, stock, self.tmicro]
        self.Q_Pricing_update(stock, R)
        
        #print(self.QPricing[self.t])

    #############################################
    ########### Learning - Forecasting ##########
    #############################################

    def V_Forecasting_update(self, stock):
        
        s = self.stateForecasting[self.traders, stock, self.tStar]
        R = self.RewardForecasting[self.traders, stock, self.tmicro, :self.NactionsForecasting].mean(axis=-1)     
        self.VForecasting[self.traders, self.tmicro, s] += self.delta_V(self.traders, R, self.VForecasting[self.traders, self.tmicro, s])  

    def Q_Forecasting_update(self, R, stock):
        
        s = self.stateForecasting[self.traders, stock, self.tStar]
        R -= ( self.VForecasting[self.traders, self.tmicro, s]*np.ones_like(R).T ).T

        Q  = self.QForecasting[self.traders, self.tmicro, s]
        self.QForecasting[self.traders, self.tmicro, s] += self.delta( self.traders, R, Q, self.actionForecastingBinary )    

    #############################################
    ############ Trading - Learning #############
    #############################################    

    def V_Trading_update(self, traders, stock, R):
         
        s0 = self.stateTrading[traders, stock, self.tmicro, 0]
        s1 = self.stateTrading[traders, stock, self.tmicro, 1]
        
        NoVolumeBinary = (s1==0)+(s1==4)
        VolumeBinary = 1-NoVolumeBinary
        
        #R = np.squeeze(R[:])
        
        R = R.mean(axis=-1)*VolumeBinary + R[:, :2].mean(axis=-1)*NoVolumeBinary #could not sell, so removing 0s from selling action for agents with no volume
        
        self.VTrading[traders, self.tmicro, s0, s1] += self.delta_V(traders, R, self.VTrading[traders, self.tmicro, s0, s1])        
                        
    def Q_Trading_update(self, traders, stock, R):

        s0 = self.stateTrading[traders, stock, self.tmicro, 0]
        s1 = self.stateTrading[traders, stock, self.tmicro, 1]

        R -= ( self.VTrading[traders, self.tmicro, s0, s1]*np.ones_like(R).T ).T

        Q = self.QTrading[traders, self.tmicro, s0, s1]
        #self.QTrading[traders, self.tmicro, s0, s1] += self.delta(traders, R, Q, self.actionTradingBinary[traders], trade=True )
        self.QTrading[traders, self.tmicro, s0, s1] += self.delta_trading(traders, R, Q)

    #############################################
    ############ Pricing - Learning #############
    #############################################
    
    def V_Pricing_update(self, stock):

        BidorAsk = self.statePricing[self.TradingNow, stock, self.tmicro, 0]
        s1 = self.statePricing[self.TradingNow, stock, self.tmicro, 1]

        R = self.RewardPricing[self.TradingNow, stock, self.tmicro].mean(axis=-1)
        
        self.VPricing[self.TradingNow, self.tmicro, BidorAsk, s1] = self.delta_V(self.TradingNow, R, self.VPricing[self.TradingNow, self.tmicro, BidorAsk, s1] )

    def Q_Pricing_update(self, stock, R):
        
        breakpoint() if (self.actionTrading[self.TradingNow, stock, self.tmicro] == 1).sum() > 0 else None #holders and waiters cannot choose, as state s0 cannot be determined
        
        BidorAsk = self.statePricing[self.TradingNow, stock, self.tmicro, 0]
        s1 = self.statePricing[self.TradingNow, stock, self.tmicro, 1] 
              
        R -= ( self.VPricing[self.TradingNow, self.tmicro, BidorAsk, s1]*np.ones_like(R).T ).T
        
        Q = self.QPricing[self.TradingNow, self.tmicro, BidorAsk, s1]
        self.QPricing[self.TradingNow, self.tmicro, BidorAsk, s1] += self.delta( self.TradingNow, R, Q, self.actionPricingBinary)

    #############################################
    ################## Rescaling ################
    #############################################

    def relative_rescaling(self, stock):
        
        self.V_Forecasting_update(stock)
        #R = self.RewardTrading[self.traders, stock, self.tmicro]
        #self.V_Trading_update(self.traders, stock, R)
        self.V_Pricing_update(stock)  

    def relative_rescaling_trading(self, stock):
        
        R = self.RewardTrading[self.traders, stock, self.tmicro]
        self.V_Trading_update(self.traders, stock, R)

   ################################
   ############ TESTS #############
   ################################

    def test_clearing(self):
    
        #breakpoint() if self.VolumeToClear == 0 and self.Price[self.stock, self.t] != self.Price[self.stock, self.tmicro-1] else None
        breakpoint() if self.Price[self.stock, self.t] == 0 else None
        #breakpoint() if self.orders_boolean('Asks', 'Market') or self.orders_boolean('Bids', 'Market') else None
 
    def test_bidaskspread(self):
        
        if self.BidAskSpread[self.stock, self.t] < 0:
        
            message = f'Warning: Negative spread {self.BidAskSpread[self.stock, self.t]}'
            self.log_message(message)
        
            message = f'CHECK order books: \n {self.OrderBookAsks}, \n {self.OrderBookBids}'
            self.log_message(message)

    def test_consistency(self, ID):
            
        IDBuyer = np.where(self.Wallet[ID, self.tBinary] < 0)[0]  
        print('Buyer Wallet:', self.Wallet[ID[IDBuyer], self.tBinary])
        print('Buyer AM:', self.AvailableMargin[ID[IDBuyer], self.tBinary])
        print('BidVolumeCleared:', self.BidOrdersToClear.Volume)
        print('Price', self.Price[self.stock, self.t])
        #print('OrderBookPayouts', OrderBookPayouts)
        print('BidOrderFees', self.BidOrdersToClear.OrderFees)
        print('BidOrdersToClear', self.BidOrdersToClear)
        self.Wallet[IDBuyer, self.tBinary] = 0 if abs(self.Wallet[ID[IDBuyer], self.tBinary]) < self.tick else breakpoint()
            
        #breakpoint() if self.OrderBookAsks.Volume.sum()+self.AvailableVolume[:, self.stock, self.t].sum(axis=0) != self.VolumeOwned[:, self.stock, tInit-1].sum() else None
        
        print('Buyer:', self.AvailableMargin[ID, self.tBinary]) if sum(self.AvailableMargin[ID, self.tBinary] < 0) > 0 else None
        breakpoint() if sum(self.AvailableMargin[ID, self.tBinary] < 0) > 0 else None

    def log_message(self, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f'[{timestamp}] {message}\n')                                
                
    ########################################
    ######## Plot Diagnotistics ############
    ########################################                 
                 
    def plot_diagnostics(self):
        
        #plt.stairs(self.AskVolumesCumulative, list(self.prices)+[self.prices[-1]*1.1], color='r'), plt.stairs(self.BidVolumesCumulative, [0]+list(self.prices), color='g'), plt.axvline(x=self.Price[self.stock, self.t], color='k', alpha=.5, linestyle='--'), plt.axhline(y=self.Volume[self.stock, self.t], color='b', alpha=.5, linestyle='--'), plt.xlim([self.prices[0], self.prices[-1]]), plt.show() # #to zoom in
        #plt.stairs(self.AskVolumesCumulative, list(self.prices)+[self.prices[-1]*1.1], color='r'), plt.stairs(self.BidVolumesCumulative, [0]+list(self.prices), color='g'), plt.axvline(x=self.Price[t, j], color='k', alpha=.5, linestyle='--'), plt.axhline(y=self.Volume[t, j], color='b', alpha=.5, linestyle='--'), plt.xlim([self.Price[t, j]-.02, self.Price[t, j]+.02]), plt.ylim([0, self.Volume[t, j]*1.2]), plt.show() # #to zoom in
        
        plt.figure(figsize=(7, 7))

        #Price dynamics
        ax1 = plt.subplot(511)
        ax1.step(self.prices, self.AskVolumesCumulative, where='post', color='r'),
        ax1.step(self.prices, self.BidVolumesCumulative, where='pre', color='g'),
        ax1.axvline(x=self.Price[self.stock, self.t], color='k', alpha=.5, linestyle='--')
        ax1.axhline(y=self.Volume[self.stock, self.t], color='b', alpha=.5, linestyle='--')
        #ax1.set_xlim([self.prices[0], self.prices[-1]]) # #to zoom in

        ax2 = plt.subplot(512)
        ax2.plot(self.Price[self.stock, self.tInit:self.t+1], color='k')
        #ax1.set_xlabel('Time step')
        ax2.set_ylabel('Stock price (€)')
        plt.tick_params('x', labelbottom=False)

        #Trading volume
        ax3 = plt.subplot(513, sharex=ax2)
        ax3.plot(self.Volume[self.stock, self.tInit:self.t+1], color='k')
        #ax2.set_xlabel('Time step')
        ax3.set_ylabel('Traded volume')

        ax4 = plt.subplot(514)
        #ax4.axvline(x=0, color='k', alpha=.5, linestyle='-')
        #ax4.scatter(self.actionT[:, 0, self.t]+((np.random.rand(self.I)-.5)*.25), self.actionForecasting[:, 0, self.t]+((np.random.rand(self.I)-.5)*.25), c=self.tau, cmap='jet', alpha=.45)
        #ax4.set_xlim([-1.1, 1.1])
        #ax4.set_ylim([-0.1, 2.1])
        #plt.colorbar(label=r'$\tau$' )#r'$g$'+' gesture param.')
        
        ax5 = plt.subplot(515)
        #ax4.axvline(x=0, color='k', alpha=.5, linestyle='-')
        ax5.scatter(self.Aequity[:, 0, self.t], self.Wallet[:, self.t], c=self.Atotal[:, self.t], cmap='jet')
        ax5.plot()
        #plt.colorba
        
        plt.show()
