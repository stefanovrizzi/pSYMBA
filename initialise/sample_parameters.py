#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:17:03 2022

@author: svrizzi
"""

import numpy as np
import pandas as pd
import random

########### Individual SYMBA Parameters #############

class ParameterSampling:
    
    def __init__(self, store, fetch, hyperparameters_to_tweak, calibrating=False):
                
        self.store = store
        self.fetch = fetch
        self.verbose = store.verbose
        store.calibrating = calibrating
        fetch.calibrating = calibrating
        self.I = fetch.input_parameters()['I']
        #self.J = fetch.input_parameters()['J']
        self.LSEstock = fetch.input_parameters()['LSEstock']
        self.CognitiveTraits = fetch.input_parameters()['CognitiveTraits']
        
        self.hyperparameters_to_tweak = hyperparameters_to_tweak
        
        randomSeed = fetch.input_parameters()['randomSeed']
        random.seed(randomSeed)
        np.random.seed(randomSeed)
        
    def tweak_hyperparameters(self, hyperparametersToTweak, hyperparameters):
        
        #for key in hyperparameters_to_tweak.keys():
        #    if key in hyperparameters.keys():
                
        for key in set(hyperparameters.keys()).intersection(hyperparametersToTweak.keys()):
            hyperparameters[key] = hyperparametersToTweak[key]
            
        #print(hyperparameters)
            
        return hyperparameters
    
    def market(self):
        
        hyperparameters={'b': 0.001,
                         'bFlat': 0,
                         'tick': 0.01,
                         'RiskFreeRate': 0.01, #0.01, #0, #0.01, #0.005,
                         'RiskPremium': 0.04, #0.04, #0, #0.04, #0.005,
                         'D': 0.02}
        
        hyperparameters = self.tweak_hyperparameters(self.hyperparameters_to_tweak, hyperparameters)  
        
        b = hyperparameters['b'] #broker fees (0.1 %)
        bFlat = hyperparameters['bFlat'] #1 € fee for trading
        tick = hyperparameters['tick'] #minimal jump in auction
        RiskFreeRate = hyperparameters['RiskFreeRate'] #(1 %)
        RiskPremium = hyperparameters['RiskPremium'] #(1 %)
        D = hyperparameters['D'] #annual stock dividend 2%
        
        ### Save parameters ###
        
        marketParameters = {'b': b,
                            'bFlat': bFlat,
                            'RiskFreeRate': RiskFreeRate,
                            'RiskPremium': RiskPremium,
                            'tick': tick,
                            'D': D}
        
        self.store.parameters(marketParameters, 'market') 
    
        
    #def wallet(self, hyperparameters_to_tweak={}, hyperparameters={'InitialVolumeOwned': 100, 'Liquidity balance': 100}):
    def wallet(self):
        
        hyperparameters={'InitialVolumeOwned': 1000,
                         'Liquidity balance': 1}
     
        hyperparameters = self.tweak_hyperparameters(self.hyperparameters_to_tweak, hyperparameters)        
        
        LiquidityBalance = hyperparameters['Liquidity balance']
        InitialVolumeOwned = hyperparameters['InitialVolumeOwned']
        
        ### Save parameters ###

        walletParameters = {'Liquidity balance': LiquidityBalance,
                            'InitialVolumeOwned': InitialVolumeOwned}
        
        self.store.parameters(walletParameters, 'wallet') 
    
    def SYMBA(self):
        
        hyperparameters={'tauPower': 2, #1.01, #2, #1.1, #2, #1.1, #2, #2.5,
                         'taumin': 5, #2, #5,
                         'taumax': 252, #21, #252,
                         'gShape': 1, #1, #2,
                         'gScale': 2, #2, #2.5,
                         'gmin': 0.01,
                         'arhoBid': 1,
                         'brhoBid': 1,
                         'rhoBidMin': 0.1, #0.05, #0.1,
                         'rhoBidMax': 0.5, #0.10, #0.2, #0.5,
                         #'kappamax': 0.01, #0.01, #1, #0.01,
                         'kappamax': 0.01, #1, #0.01, #1, #0.01,                         
                         'kappamin': 0.0005, #0.01, #0.1, #0.0005,
                         'etaHalf': 1, #0.01, #0.5,
                         'nu_plus_d': 0.50,
                         'nu_minus_d': 0.50,
                         'nu_zero_d': 0.05,
                         'sigma': 0.1} #0.3
        
        hyperparameters = self.tweak_hyperparameters(self.hyperparameters_to_tweak, hyperparameters)        
        
        Tw, Tm, Ty = self.fetch.time_references()
        
        ### TAU ### investment horizon        
        #tau = np.random.power(hyperparameters['tauPower'], size=self.I) #large tau more frequent

        def power_law_sample(T1, T2, alpha, size):
            # Generate uniform random numbers
            u = np.random.uniform(0, 1, size)
    
            # Inverse CDF of the power-law distribution
            result = ((u * (T2**(1-alpha) - T1**(1-alpha)) + T1**(1-alpha)) ** (1 / (1-alpha)))
    
            return result

        # Example usage
        #T1 = hyperparameters['taumin']  # lower bound
        #T2 = hyperparameters['taumax'] # upper bound
        #alpha = hyperparameters['tauPower']  # power-law exponent

        #tau = power_law_sample(T1, T2, alpha, size=self.I)  # Generate 1000 samples
        
        #Uniform
        tau = np.random.uniform(hyperparameters['taumin'], hyperparameters['taumax'], size=self.I)
        
        #tau *= (hyperparameters['taumax']-hyperparameters['taumin']) #with upperbound tau sampling for uniform distribution
        #tau += hyperparameters['taumin']
        tau = tau.astype(int)

        #tau += Tw
        #tmax = max(tau)
        
        print('tau max: ', max(tau))
        
        tau = np.sort(tau)
        
        tInit = max(Ty+1, tau[-1]+1)
        
        ### KAPPA ### Knowledge of true fundamental value x
        LSE = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)
        LSEstock = self.fetch.input_parameters()['LSEstock']
        PriceLSE = LSE[LSEstock].round(2) 
        returnsLSE = PriceLSE.diff().iloc[1:]
        
        riskFreeRate = self.fetch.parameters_json('market')['RiskFreeRate']
        riskPremium = self.fetch.parameters_json('market')['RiskPremium']
        
        mu = riskFreeRate+riskPremium #annual growth rate
        sigma = hyperparameters['sigma'] #max(hyperparameters['sigma'], mu*2) #annual volatility rate

        kappamin = hyperparameters['kappamin']
        kappamax = hyperparameters['kappamax']
        kappa = np.random.rand(self.I)*(kappamax-kappamin)+kappamin

        #import matplotlib.pyplot as plt
        #plt.hist(tau), plt.show()
        #plt.hist(kappa), plt.show()        
        #kappa = 1/tau
        
        #kappa = np.random.normal(0.01, 0.07, size=self.I)
        #kappa = abs(kappa)
        
        #kappa = tau/max(tau)*(kappamax-kappamin)+kappamin
        
        ### RHO ### Risk parameter
        rhoAsk = np.ones(self.I) #*.5
        
        rhoBidMin = hyperparameters['rhoBidMin'] #0.1
        rhoBidMax = hyperparameters['rhoBidMax']
                
        rhoBid = np.random.beta(a=hyperparameters['arhoBid'], b=hyperparameters['brhoBid'], size=self.I) #*0.99+0.01
        rhoBid = rhoBid*(rhoBidMax-rhoBidMin)+rhoBidMin
        
        walletParameters = self.fetch.parameters_json('wallet')
        InitialVolumeOwned = walletParameters['InitialVolumeOwned']
        LiquidityBalance = walletParameters['Liquidity balance']
        PriceOverWealth = 1/ (InitialVolumeOwned*(1+LiquidityBalance) )
          
        b = self.fetch.parameters_json('market')['b']
        
        print('\n ** WARNING: low rhoAM ** \n') if sum(rhoBid < PriceOverWealth*(1+2*b)) > 0 else None
        rhoBid = np.maximum(rhoBid, PriceOverWealth*(1+2*b))

        ### Nu ### Normalised (0-1) percentage of sign reversal in forecast reward        
        nu_plus_d = hyperparameters['nu_plus_d']
        nu_minus_d = hyperparameters['nu_minus_d']
        nu_zero_d = hyperparameters['nu_zero_d']
        
        ### ETA ### Pricing reward decay
        etaHalf = hyperparameters['etaHalf'] #parameter for exponential decay in pricing return for limit orders (/ % of miss to get half reward)
        
        #For trading reward
        rewardMultiplier = 1
        
        ### G ### Transaction gesture
        
        g = np.random.gamma(shape=hyperparameters['gShape'], scale=hyperparameters['gScale'], size=self.I)/100
        g += hyperparameters['gmin'] #if sum(g < 0.01) > 0 else 0
        #g = np.random.uniform(0.01, 0.05, size=self.I)
        #g = np.random.uniform(0.01, 0.10, size=self.I)
                        
        breakpoint() if sum(g<=0) > 0 else None
        
        gSkewness = np.random.choice([-1, 1], size=self.I)
        
        #print(hyperparameters['gShape'], hyperparameters['gScale'])
        
        #import matplotlib.pyplot as plt
        
        #plt.hist(rhoAvailableMargin, bins=np.linspace(0, 1, 21)), plt.show()
        #plt.hist(tau), plt.show()
        #plt.hist(g), plt.title(str(min(g))+' '+str(max(g))), plt.show()
        #plt.hist(tau), plt.title(str(min(tau))+' '+str(max(tau))), plt.show()
        ##g = np.random.gamma(shape=6, scale=10, size=self.I)/100
        #breakpoint()
        
        # the bounds of this uniform distribution interval are selected
        #according to one of the model’s hyperparameters, namely the gesture scalar ζi
        
        DrawdownLimit =  50*np.ones(self.I) #( np.random.rand(self.I)*5 ) + 25 #Do we need to replace bankrupt agents ? Simply stop to participate
        #Originally, 40%-50%. If the year-to-date, peak-to-bottom, absolute percentage decrease of the agent’s portfolio net asset value
        # exceeds this drawdown limit li at any time step t, then the agent is listed
        #as bankrupt and unable to interact with the market anymore. These value
        #may seem unrealistic in the sense that the drawdowns commonly tolerated
        #in the asset management industry rarely exceed 30%. However, contrarily
        #to a real stock exchange, such a large value is a necessity due to the model
        #relying on a constant population of agents, which are not replaced if they become bankrupt.
        
        data ={'rhoAsk': rhoAsk,
        'rhoBid': rhoBid,
        'tau': tau,
        'g': g,
        'gSkewness': gSkewness,
        'kappa': kappa}
        
        df = pd.DataFrame(data)
        df.to_csv(self.store.DataFolderPath+self.store.example_folder_path_name()+'/SYMBA_parameters.csv')
        
        ### Save parameters ###
            
        SYMBAParameters = {'mu': mu,
                           'sigma': sigma,
                           'kappamin': kappamin,
                           'kappamax': kappamax,
                           'etaHalf': etaHalf,
                           'nu_plus_d': nu_plus_d,
                           'nu_minus_d': nu_minus_d,
                           'nu_zero_d': nu_zero_d,
                           'tauPower': hyperparameters['tauPower'],
                           'taumin': hyperparameters['taumin'],
                           'taumax': hyperparameters['taumax'],
                           'gShape': hyperparameters['gShape'],
                           'gScale': hyperparameters['gScale'],
                           'arhoBid': hyperparameters['arhoBid'],
                           'brhoBid': hyperparameters['brhoBid'],
                           'rhoBidMax': rhoBidMax,
                           'rhoBidMin': rhoBidMin,
                           'rewardMultiplier': rewardMultiplier,
                           'tInit': tInit}
        
        self.store.parameters(SYMBAParameters, 'SYMBA') 

    ########### RL Q-learning Parameters #############
    
    #def RL(self, CognitiveTraits, hyperparameters_to_tweak={}, hyperparameters={'aAlphaConf':2.6, 'bAlphaConf':1.1, 'aAlphaDisc':1.1, 'bAlphaDisc':2.6, 'distribution':True, 'kernel':'gaussian', 'alphaConf':0.03, 'alphaDisc':0.01, 'beta':1}): #'alphaConf':0.3, 'alphaDisc':0.2,

    def RL(self):
        
        #Default
        hyperparameters={'aAlphaConf':1.5,
                         'bAlphaConf':1.5,
                         'aAlphaDisc':1.5,
                         'bAlphaDisc':1.5,
                         'distribution':True,
                         'kernel':None,
                         'alphaConf':0.3,
                         'alphaDisc':0.1,
                         'beta':1,
                         'alphaConfMax': .5,
                         'alphaDiscMax': .5}
        
        #Tweak hyperparameters        
        hyperparameters = self.tweak_hyperparameters(self.hyperparameters_to_tweak, hyperparameters)
        
        if len( self.hyperparameters_to_tweak.keys() ) == 0:
        
            if 'no bias' in self.CognitiveTraits:
        
                hyperparameters['alphaConf'] = hyperparameters['alphaDisc']
        
                hyperparameters['aAlphaConf'] = 1.5
                hyperparameters['aAlphaDisc'] = 1.5
                hyperparameters['bAlphaConf'] = 1.5
                hyperparameters['bAlphaDisc'] = 1.5  

            elif 'optimistic bias' in self.CognitiveTraits:
         
                hyperparameters['alphaConf'] = 0.3
                hyperparameters['alphaDisc'] = 0.1
         
                hyperparameters['aAlphaConf'] = 2
                hyperparameters['bAlphaConf'] = 1
                hyperparameters['aAlphaDisc'] = 1                
                hyperparameters['bAlphaDisc'] = 2  

            elif 'pessimistic bias' in self.CognitiveTraits:

                hyperparameters['alphaConf'] = 0.1
                hyperparameters['alphaDisc'] = 0.3
         
                hyperparameters['aAlphaConf'] = 1
                hyperparameters['bAlphaConf'] = 2
                hyperparameters['aAlphaDisc'] = 2                
                hyperparameters['bAlphaDisc'] = 1
                
            elif 'no learning' in self.CognitiveTraits:
            
                hyperparameters['alphaConfMax'] = 0
                hyperparameters['alphaDiscMax'] = 0

    #if 'optimistic bias' in cognitivetrait:
        
    #    hyperparameters_to_tweak={'aAlphaConf':2, 'bAlphaConf':1, 'aAlphaDisc':1, 'bAlphaDisc':2}
        
    #elif 'no bias' in cognitivetrait or 'no bias individual' in cognitivetrait:
        
    #    hyperparameters_to_tweak={'aAlphaConf':1.5, 'bAlphaConf':1.5, 'aAlphaDisc':1.5, 'bAlphaDisc':1.5}
            
    #elif 'pessimistic bias' in cognitivetrait:
        
    #    hyperparameters_to_tweak={'aAlphaConf':1, 'bAlphaConf':2, 'aAlphaDisc':2, 'bAlphaDisc':1}
        
    #elif 'no learning' in cognitivetrait:
        
    #    hyperparameters_to_tweak={'alphaConfMax': 0, 'alphaDiscMax': 0}

        
        if hyperparameters['distribution']:
            
            if hyperparameters['kernel']:
                
                betaDistribution, alphaConfDistribution, alphaDiscDistribution = self.fetch.RL_parameters_data('ASYMMETRIC RELATIVE')
                
                kernel = 'gaussian'
                print(f'Sampling RL parameters from {kernel} kernelised parameter distribution fitted on RL experiment data') if self.verbose else None
                
                from sklearn.neighbors import KernelDensity
                
                #With covariance matrix between parameteres
                #X = np.zeros((len(betaDistribution), 3))
                #X[:, 0] = betaDistribution
                #X[:, 1] = alphaConfDistribution
                #X[:, 2] = alphaDiscDistribution
                #kde = KernelDensity(kernel=kernel, bandwidth=0.2).fit(X)
                #SyntheticRLparameters = kde.sample( self.I )
                
                #Independent parameters (covariance-less)                
                kde = KernelDensity(kernel=kernel, bandwidth=0.2).fit(betaDistribution.reshape(-1, 1))
                beta = 1 #np.squeeze( kde.sample( self.I ).T )

                kde = KernelDensity(kernel=kernel, bandwidth=0.2).fit(alphaConfDistribution.reshape(-1, 1))
                #alphaConf = np.random.uniform(0.0005, 0.003, size=self.I) #np.squeeze( kde.sample( self.I ).T )
                alphaConf = np.squeeze( kde.sample( self.I ).T )

                kde = KernelDensity(kernel=kernel, bandwidth=0.2).fit(alphaDiscDistribution.reshape(-1, 1))
                #alphaDisc = np.random.uniform(0.0005, 0.001, size=self.I) #np.squeeze( kde.sample( self.I ).T )
                alphaDisc = np.squeeze( kde.sample( self.I ).T )
                
                #Initialise parameters with values sampled from kernel density distribution
                beta = abs( beta )
                
                alphaConf = abs( alphaConf )
                #alphaConf /= 10
                alphaConf = np.maximum(alphaConf, 0.01)
                alphaConf = alphaConf / max(alphaConf) * (alphaConf > 1) + alphaConf * (alphaConf <= 1)
                                
                alphaDisc = abs( alphaDisc ) 
                #alphaDisc /= 10
                alphaDisc = np.maximum(alphaDisc, 0.01)  
                alphaDisc = alphaDisc / max(alphaDisc) * (alphaDisc > 1) + alphaDisc * (alphaDisc <= 1)
                
            elif not hyperparameters['kernel']:
            
                beta = np.ones(self.I)*hyperparameters['beta']
                alphaConf = np.random.beta(a=hyperparameters['aAlphaConf'], b=hyperparameters['bAlphaConf'], size=self.I)
                #alphaConf = alphaConf * hyperparameters['alphaConf'] #shorten range to lower alphas
                
                alphaDisc = np.random.beta(a=hyperparameters['aAlphaDisc'], b=hyperparameters['bAlphaDisc'], size=self.I)
                #alphaDisc = alphaDisc * hyperparameters['alphaDisc'] #shorten range to lower alphas
                                                 
        elif not hyperparameters['distribution']: #no distribution, every trader same value
                    
            beta = np.ones(self.I)*hyperparameters['beta']
            alphaConf = np.ones(self.I)*hyperparameters['alphaConf']
            alphaDisc = np.ones(self.I)*hyperparameters['alphaDisc']
        
            #import matplotlib.pyplot as plt
            #plt.hist(alphaConf, label=r'$\alpha_{+}$')
            #plt.hist(alphaDisc, label=r'$\alpha_{-}$')
            #plt.legend()
            #plt.show()
            
            #plt.scatter(alphaConf, alphaDisc), plt.xlabel(r'$\alpha_{+}$'), plt.ylabel(r'$\alpha_{-}$'), plt.show()

        #alphaConf = alphaDisc if 'optimistic bias' not in CognitiveTraits else alphaConf

        # Adjust alpha range
        alphaConf *= hyperparameters['alphaConfMax']
        alphaDisc *= hyperparameters['alphaDiscMax']


        if 'no bias individual' in self.CognitiveTraits:
            alphaDisc = alphaConf
        
        alphaV = (alphaConf + alphaDisc)/2 #mean
        
        print(self.CognitiveTraits, alphaConf.mean(), alphaDisc.mean(), alphaV.mean(), '\n')

        keys = ['beta', 'alphaConf', 'alphaDisc', 'alphaV']
        values = [beta, alphaConf, alphaDisc, alphaV]
    
        RLParameters = dict(zip(keys, values))
    
        #self.store.parameters(RLParameters, 'RL')
        #import matplotlib.pyplot as plt
        #plt.scatter(alphaConf, alphaDisc), plt.xlabel(r'$\alpha_{+}$'), plt.ylabel(r'$\alpha_{-}$'), plt.show()
        
        df = pd.DataFrame(RLParameters)
        df.to_csv(self.store.DataFolderPath+self.store.example_folder_path_name()+'/RL_parameters.csv')
        
        
        ### Save parameters ###
            
        RLHyperParameters = {'aAlphaConf':hyperparameters['aAlphaConf'],
                             'bAlphaConf':hyperparameters['bAlphaConf'],
                             'aAlphaDisc':hyperparameters['aAlphaDisc'],
                             'bAlphaDisc':hyperparameters['bAlphaDisc'],
                             'alphaConfMax':hyperparameters['alphaConfMax'],
                             'alphaDiscMax':hyperparameters['alphaDiscMax'],
                             'distribution':hyperparameters['distribution'],
                             'kernel':hyperparameters['kernel'],
                             'alphaConf':hyperparameters['alphaConf'],
                             'alphaDisc':hyperparameters['alphaDisc'],
                             'beta':hyperparameters['beta']} #,
                             #'alphamax': hyperparameters['alphamax']}
        
        self.store.parameters(RLHyperParameters, 'RL')         
        
