#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:57:58 2023

@author: svrizzi
"""

# importing matplot lib
import matplotlib.pyplot as plt
import numpy as np
 
from initialise.time_parameters import tInit

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching
import pandas as pd

# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
 
import matplotlib

# numpy array
#x = np.linspace(-2, 2, 200)

n = 0

class TrainingData:

    def __init__(self, fetch):
        
        self.fetch = fetch

        #Input variables
        self.InputParameters = fetch.input_parameters()
        self.T = self.InputParameters['T']
        self.I = self.InputParameters['I']
        self.J = self.InputParameters['J']
        self.S = self.InputParameters['S']

        #RL parameters
        self.RLparameters = fetch.parameters('RL')
        self.CognitiveTraits = self.InputParameters['CognitiveTraits']
        
        self.beta = self.RLparameters['beta']
        self.alphaConf = self.RLparameters['alphaConf']
        
        if 'optimistic bias' in self.CognitiveTraits:
            self.alphaDisc = self.RLparameters['alphaDisc']
        
        
        #SYMBA parameters
        self.SYMBAparameters = fetch.parameters('SYMBA')
        self.FundamentalValues = self.SYMBAparameters['FundamentalValues'][:, 0] #stock 0
        #self.rho = self.SYMBAparameters['rho']
        self.tau = self.SYMBAparameters['tau']
        self.g = self.SYMBAparameters['g']
        
        #Data generated from simulation
        #self.data = fetch.saved_data(s)
        
        #Market variables
        self.Aequity = np.zeros((self.I, self.T-tInit+1))
        self.Abonds = np.zeros((self.I, self.T-tInit+1))
        self.Qs = np.zeros((self.I, self.T-tInit+1))
        self.P = np.zeros((self.T-tInit))
    
        #for s in range(self.S):
        #    self.A_bonds[s] = fetch.saved('A_bonds', s=s, verbose=False)[0, tInit:]
        #    self.A_equity[s] = fetch.saved('A_equity', s=s, verbose=False)[0, tInit:]

        self.Abonds = fetch.saved('Abonds', s=n, verbose=False)[:, tInit-1:]
        self.Aequity = fetch.saved('Aequity', s=n, verbose=False)[:, 0, tInit-1:]
        self.Qs = fetch.saved('Qs', s=n, verbose=False)[:, 0, tInit-1:]
        self.P = fetch.saved('P', s=n, verbose=False)[0, tInit-1:]

N_ready = 30

store = FileStorage()
store.set_example(0)
fetch = FileFetching(store)
td = TrainingData(fetch)

Abonds = np.zeros((N_ready, td.I, td.T-tInit+1))
Aequity = np.zeros((N_ready, td.I, td.T-tInit+1))
P = np.zeros((N_ready, td.T-tInit+1))
Qs = np.zeros((N_ready, td.I, td.T-tInit+1))
g_agents = np.zeros((N_ready, td.I))
#tau_agents = np.zeros((N_ready*20, td.I))
FV = np.zeros((N_ready, td.I))

for exampleNumber in range(N_ready): #range(506):

    store = FileStorage()
    store.set_example(exampleNumber)
    fetch = FileFetching(store)
    
    td = TrainingData(fetch)

    Abonds[exampleNumber] = td.Abonds
    Aequity[exampleNumber]  = td.Aequity
    Qs[exampleNumber] = td.Qs
    P[exampleNumber] = td.P #sim
    g_agents[exampleNumber] = td.g
    FV[exampleNumber] = td.FundamentalValues
   
    #tau_agents[20*exampleNumber:20*exampleNumber+20] = td.tau
   
g_agents = g_agents*100

# duration of the video
duration = (td.T-tInit+1)/3
fps = 24

cmap_custom = matplotlib.colors.ListedColormap(['r', 'k', [0, 1, .3]])

for exampleNumber in [0, 29]:

    rich_limit = Aequity[exampleNumber, 0, 0]+Abonds[exampleNumber, 0, 0]
    x = np.linspace(0, rich_limit, 1000)[::-1]
    y = [rich_limit-x_i for x_i in x]
    # method to get frames
    
    # matplot subplot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    #fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[30, 1]}, figsize=(6, 6), dpi=300)
    #cax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    #axs = axs.ravel()
    #ax = axs[0]
    #axcb = axs[1]
    
    def scatter_plot_wealth(t, ax):
        
        #print(t)
        # plotting line
        ax.plot(x, y,'k--', alpha=.2, lw=0.7, label='Initial total wealth') #richness line
        ax.scatter( Abonds[exampleNumber, :, t], Aequity[exampleNumber, :, t],
                   c = g_agents[exampleNumber],
                   s=2.5, cmap='turbo', alpha=.5)
        #ax.colorbar(sc)
        #fig.colorbar(sc, ax=axcb)
        
        #fig.colorbar(sc, ax=ax, cax=axcb)
        #fig.subplots_adjust(wspace=0.1, hspace=0.5)
        
        ax.set_xlabel('Av. margin ($)')
        ax.set_ylabel('Stock value owned ($)')
        ax.set_xscale('log')
        #ax.set_yscale('log')
        #ax.loglog()
        ax.set_xlim([1, 30000])
        ax.set_ylim([-500, 30000])
        
    def scatter_plot_parameters(t, ax):
        
        #print(t)
        # plotting line
        
        if t == 0:
            c = ['k']*td.I
        else:
            g = (Qs[exampleNumber, :, t]-Qs[exampleNumber, :, t-1])>0 #bought stocks
            r = ((Qs[exampleNumber, :, t]-Qs[exampleNumber, :, t-1])<0)*(-1) #sold stocks
            c = g+r
            
        ax.axvline(x=P[exampleNumber, t], color='k', alpha=.2, lw=0.7, label='Price') #richness line
        ax.scatter( FV[exampleNumber, :], g_agents[exampleNumber, :], c = c, cmap=cmap_custom,
                   s=np.log10(Qs[exampleNumber, :, t]+10))

        ax.set_xlabel('Fundamental Values ($)')
        ax.set_ylabel(r'$g$ (% of stock price)')
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        #ax.loglog()
        #ax.set_xlim([1, 30000])
        #ax.set_ylim([-500, 30000])
    
    
    def make_frame(t):
         
        # clear
        ax.clear()
        #axcb.clear()
        
        t = int(t*3) #t = 1/fps
        
        #scatter_plot_wealth(t, ax)
        scatter_plot_parameters(t, ax)
        
        ax.legend()
        ax.set_title(f'Time step: {t}')
        
        #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        #ax.color 
        
        # returning numpy image
        return mplfig_to_npimage(fig)
     
       
    # creating animation
    animation = VideoClip(make_frame, duration = duration)

    #animation.write_videofile(f"my_animation_exN_{exampleNumber}.mp4", fps=fps)
    animation.write_videofile(f"my_parameters_exN_{exampleNumber}.mp4", fps=fps)

#breakpoint() 

#animation.save('movie.mp4')

# displaying animation with auto play and looping
#animation.ipython_display(fps = 20, loop = True, autoplay = True)
