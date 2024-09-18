#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:46:49 2022

@author: svrizzi
"""

import sys
import os
import matplotlib.pyplot as plt

from pyparsing import line
sys.path.append('..') if '..' not in sys.path else None  #go to parent folder
import numpy as np

sliding_window_view = np.lib.stride_tricks.sliding_window_view

class PlotPretraining:

    def __init__(self, fetch):
        
        self.fetch = fetch

        self.InputParameters = self.fetch.input_parameters()
        self.InputTimeParameters = self.fetch.input_parameters(time=True)
        self.trader_parameters()        
        self.input_parameters()

        self.create_plot_folders()
        self.RL_variables()
        
        del self.fetch
        
        #breakpoint()

    def create_plot_folders(self):

        self.example_folder_path = self.fetch.example_folder_path

        #Save plots in folders
        self.plotFolderPath = self.fetch.plotFolderPath

        #Create folders
        try:
            os.mkdir(self.plotFolderPath) if not os.path.isdir(self.plotFolderPath) else None #create Data folder for all data files, in case it does not exist
        except:
            self.plotFolderPath += '../'
            os.mkdir(self.plotFolderPath) if not os.path.isdir(self.plotFolderPath) else None #create Data folder for all data files, in case it does not exist

        self.plotPath = self.plotFolderPath+self.example_folder_path()
        os.mkdir(self.plotPath) if not os.path.isdir(self.plotPath) else None #create Data folder for all data files, in case it does not exist

        self.plotPath = self.plotFolderPath+self.example_folder_path()+'/pretraining'
        os.mkdir(self.plotPath) if not os.path.isdir(self.plotPath) else None #create Data folder for all data files, in case it does not exist
        
    def input_parameters(self):

        #Input parameters
        self.Tbatch = self.InputTimeParameters['TbatchPretraining']
        self.tInit = self.fetch.input_parameters_json('SYMBA')['tInit']
        self.Tbatch -= self.tInit
        self.Tbatch -= max(self.tau)
        self.I = self.InputParameters['I']
        self.J = self.InputParameters['J']
        self.S = self.InputTimeParameters['Spretraining']
        self.T = int( self.Tbatch *self.S)    

    def trader_parameters(self):

        #RL parameters
        self.RLparameters = self.fetch.parameters('RL')
        self.CognitiveTraits = self.InputParameters['CognitiveTraits']
        
        self.beta = self.RLparameters['beta']
        self.alphaConf = self.RLparameters['alphaConf']
        
        if 'optimistic bias' in self.CognitiveTraits:
            self.alphaDisc = self.RLparameters['alphaDisc']
        
        #SYMBA parameters
        self.SYMBAparameters = self.fetch.parameters('SYMBA')
        self.FundamentalValues = self.SYMBAparameters['FundamentalValues'][:, 0] #stock 0
        self.rho = self.SYMBAparameters['rho']
        self.tau = self.SYMBAparameters['tau']
        self.g = self.SYMBAparameters['g']    
        
    def RL_variables(self):

       self.actionT = np.zeros((self.I, self.T))
       self.QF = np.zeros((self.I, self.T, 3))
       self.QT = np.zeros((self.I, self.T, 3, 4, 2, 3))
       self.QP = np.zeros((self.I, self.T, 2, 4, 2, 4, 3))
       self.QR = np.zeros((self.I, self.T, 2, 4, 2, 5))
    
       for s in range(self.S):
            
            #RL variables
            self.QF[:, s*self.Tbatch:(s+1)*self.Tbatch] = self.fetch.saved('QForecasting', f'pretraining_{s}')[:, self.tInit+max(self.tau):]
            self.QT[:, s*self.Tbatch:(s+1)*self.Tbatch] = self.fetch.saved('QTrading', f'pretraining_{s}')[:, self.tInit+max(self.tau):]
            self.QP[:, s*self.Tbatch:(s+1)*self.Tbatch] = self.fetch.saved('QPricing', f'pretraining_{s}')[:, self.tInit+max(self.tau):]
            self.QR[:, s*self.Tbatch:(s+1)*self.Tbatch] = self.fetch.saved('QReferencing', f'pretraining_{s}')[:, self.tInit+max(self.tau):]
            
            
    ################################
    ############# Plots ############
    ################################
    
    def QF_dynamics(self):

        self.QFfolder = self.plotPath+'/QF'
        os.mkdir(self.QFfolder) if os.path.isdir(self.QFfolder) == False else None #create Data folder for all data files, in case it does not exist
    
        colours = ['k', 'r', 'b']

        traders = np.argsort(self.tau)
        n = min(25, self.I)            
        n_blocks = int(self.I/n)

        for block in range( n_blocks ):
        
            traders_temp = range( block*n, (block+1)*n ) 

            fig, axs = plt.subplots(int(np.sqrt(n)), int(np.sqrt(n)), figsize=(9, 9), sharex=True, sharey=True)
            axs = axs.ravel()
        
            for i, trader in enumerate(traders_temp):
                [axs[i].plot(self.QF[trader, :, a], alpha=.5, color=colours[a]) for a in range(np.shape(self.QF)[-1])]
  
            title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(self.QFfolder+f'/QF - traders {traders_temp[0]}-{traders_temp[-1]}.png', dpi=200)
            plt.close()
        
        return

    def QT_dynamics(self):
        
        self.QTfolder = self.plotPath+'/QT'
        os.mkdir(self.QTfolder) if os.path.isdir(self.QTfolder) == False else None #create Data folder for all data files, in case it does not exist
                
        colours = [[0, .5, 0], 'k', [.5, 0, 0]]
        
        traders = np.argsort(self.tau)
        n = min(25, self.I)            
        n_blocks = int(self.I/n)
        
        for s0 in range(np.shape(self.QT)[2]):
            for s1 in range(np.shape(self.QT)[3]):
                for s2 in range(np.shape(self.QT)[4]):
                                            
                    for block in range( n_blocks ):
                                
                        traders_temp = range( block*n, (block+1)*n ) 
                                
                        fig, axs = plt.subplots(int(np.sqrt(n)), int(np.sqrt(n)), figsize=(9, 9), sharex=True, sharey=True)
                        axs = axs.ravel()
                        
                        for i, trader in enumerate(traders_temp):
                            [axs[i].plot(self.QT[trader, :, s0, s1, s2, a], alpha=.8, color=colours[a]) for a in range(np.shape(self.QT)[-1])]              
                
                        title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
                        plt.suptitle(title)
                        plt.tight_layout()
                        plt.savefig(self.QTfolder+f'/QT training - {s0, s1, s2} - traders {traders_temp[0]}-{traders_temp[-1]}.png', dpi=200)
                        plt.close()
        
        return
    
    def QP_dynamics(self):
    
        self.QPfolder = self.plotPath+'/QP'
        os.mkdir(self.QPfolder) if os.path.isdir(self.QPfolder) == False else None #create Data folder for all data files, in case it does not exist
        
        colours = [[.5, 0, 0], [0, .4, 0], [0, 0, .5]]
        
        traders = np.argsort(self.g)
        n = min(25, self.I)            
        n_blocks = int(self.I/n)
        
        for s0 in range(np.shape(self.QP)[2]):
            for s1 in range(np.shape(self.QP)[3]):
                for s2 in range(np.shape(self.QP)[4]):
                    for s3 in range(np.shape(self.QP)[5]): 
                        
                        for block in range( n_blocks ):
        
                            traders_temp = range( block*n, (block+1)*n ) 
                        
                            fig, axs = plt.subplots(int(np.sqrt(n)), int(np.sqrt(n)), figsize=(9, 9), sharex=True, sharey=True)
                            axs = axs.ravel()
                        
                            for i, trader in enumerate(traders_temp):
                                [axs[i].plot(self.QP[trader, :, s0, s1, s2, s3, a], alpha=.5, color=colours[a]) for a in range(np.shape(self.QP)[-1])]              
                
                            title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
                            plt.suptitle(title)
                            plt.tight_layout()
                            plt.savefig(self.QPfolder+f'/QP training - {s0, s1, s2, s3} - traders {traders_temp[0]}-{traders_temp[-1]}.png', dpi=200)
                            plt.close()
        
        return

    def QR_dynamics(self):
        
        self.QRfolder = self.plotPath+'/QR'
        os.mkdir(self.QRfolder) if os.path.isdir(self.QRfolder) == False else None #create Data folder for all data files, in case it does not exist
        
        colours = [[.5, 0, 0], [0, .4, 0], [0, 0, .5], [.5, 0, .5], 'k']
        
        traders = np.argsort(self.g)
        n = min(25, self.I)            
        n_blocks = int(self.I/n)
        
        for s0 in range(np.shape(self.QR)[2]):
            for s1 in range(np.shape(self.QR)[3]):
                for s2 in range(np.shape(self.QR)[4]):
                               
                    for block in range( n_blocks ):
        
                        traders_temp = range( block*n, (block+1)*n ) 
                        
                        fig, axs = plt.subplots(int(np.sqrt(n)), int(np.sqrt(n))+1, figsize=(9, 9), sharex=True, sharey=True)
                        axs = axs.ravel()
                        
                        for i, trader in enumerate(traders_temp):
                            [axs[i].plot(self.QR[trader, :, s0, s1, s2, a], alpha=.5, color=colours[a]) for a in range(np.shape(self.QR)[-1])]               
                
                        title = '' #f'Trader: {i}' #f'Hyperparameter of interest: {self.hyperparameterName}'
                        plt.suptitle(title)
                        plt.tight_layout()
                        plt.savefig(self.QRfolder+f'/QR training - {s0, s1, s2} - traders {traders_temp[0]}-{traders_temp[-1]}.png', dpi=200)
                        plt.close()
        
        return
