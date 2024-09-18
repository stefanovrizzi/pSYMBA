#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:01:14 2023

@author: svrizzi
"""

################ Import key functions ############################

#import sys
#sys.path.append('..') if '..' not in sys.path else None  #go to parent folder
from plot.examples import PlotExamples
from plot.confirmation_bias import PlotConfirmationBias

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching

class ExampleAnalysis:
    
    def __init__(self, datasets):
        
        #loop over datasets
        store = FileStorage()
        fetch = FileFetching(store)
        
        ##datasetFolders =  fetch.all_dataset_folders() if datasets == 'all' else [datasets]
        #datasetFolders = fetch.all_dataset_folders('arhoAvailableMargin-1-10 bias OFF')
        #[PlotExamples(datasetFolder) for datasetFolder in datasetFolders] #can parallelise

        #breakpoint()
        
        #datasetFoldersConfirmationBiasAnalysis = fetch.all_dataset_folders('ConfirmationBiasAnalysis')
        datasetFoldersConfirmationBiasAnalysis = ['ConfirmationBiasAnalysis alphaDisc-0.0-0.9 alphaConf-0.0-0.9 distribution-False kernel-False bFlat 1 - beta 1']        

        [PlotConfirmationBias(datasetFolder) for datasetFolder in datasetFoldersConfirmationBiasAnalysis] #can parallelise
            

def get_parser():
    
    #https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='?', const=True, type=str, help = 'please give the name of a dataset folder (i.e. notes given to run training_data.py)', required=False, default='all')
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    args = get_parser()

    ExampleAnalysis(args.datasets)
