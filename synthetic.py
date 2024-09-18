#%%Import libraries

import numpy as np
import pandas as pd

#%% 

class Synthetic:
    
    def __init__(self, df, pool_exp, model, master_datafile, export, folder, filename, simulations=False):
        
        #%% Optimisation process

        '''
        REINFORCEMENT LEARNING MODEL FITTING
        '''
        
        #Whether or not to pool data from two experiments (test and retest)
        self.pool_exp = pool_exp #simulations fit independently, but come from parameters fit on pooled experiments
        
        #Data   
        # Fitting across experiments (pool_exp = True) or individual experiments
        self.df = df.reset_index().set_index('id') if pool_exp else df 
        
        self.subjects = master_datafile['subjects']
        
        self.context_dict = master_datafile['context_dict']
    
        #%% Models of choice
        
        self.model = model
                
        self.export = export
        self.folder_parameters = folder
        self.simulations = simulations
    
        self.filename = filename
