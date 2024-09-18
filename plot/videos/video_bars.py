#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 22:39:05 2023

@author: svrizzi
"""

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
 
# numpy array
#x = np.linspace(-2, 2, 200)

import matplotlib

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False

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
        #self.FundamentalValues = self.SYMBAparameters['FundamentalValues'][:, 0] #stock 0
        #self.rho = self.SYMBAparameters['rho']
        self.tau = self.SYMBAparameters['tau']
        self.g = self.SYMBAparameters['g']
        
        #Data generated from simulation
        #self.data = fetch.saved_data(s)
        
        #Market variables
        self.Aequity = np.zeros((self.I, self.T-tInit+1))
        self.Abonds = np.zeros((self.I, self.T-tInit+1))
    
        #for s in range(self.S):
        #    self.A_bonds[s] = fetch.saved('A_bonds', s=s, verbose=False)[0, tInit:]
        #    self.A_equity[s] = fetch.saved('A_equity', s=s, verbose=False)[0, tInit:]

        self.Abonds = fetch.saved('Abonds', s=n, verbose=False)[:, tInit-1:]
        self.Aequity = fetch.saved('Aequity', s=n, verbose=False)[:, 0, tInit-1:]


N_ready = 30

store = FileStorage()
store.set_example(0)
fetch = FileFetching(store)
td = TrainingData(fetch)

Abonds = np.zeros((N_ready, td.I, td.T-tInit+1))
Aequity = np.zeros((N_ready, td.I, td.T-tInit+1))
#P = np.zeros((N_ready, td.T-tInit))
g_agents = np.zeros((N_ready, td.I))
#tau_agents = np.zeros((N_ready*20, td.I))

for exampleNumber in range(N_ready): #range(506):

    store = FileStorage()
    store.set_example(exampleNumber)
    fetch = FileFetching(store)
    
    td = TrainingData(fetch)

    Abonds[exampleNumber] = td.Abonds
    Aequity[exampleNumber]  = td.Aequity
    #P[exampleNumber] = td.P[n] #sim
    g_agents[exampleNumber] = td.g
   
    #tau_agents[20*exampleNumber:20*exampleNumber+20] = td.tau
   
g_agents = g_agents*100

exampleNumber = 29

# duration of the video
duration = (td.T-tInit+1)/3
fps = 24

rich_limit = Aequity[exampleNumber, 0, 0]+Abonds[exampleNumber, 0, 0]

sorted_idx = sorted(range(len(g_agents[exampleNumber])), key=lambda k: g_agents[exampleNumber, k])

# matplot subplot
fig, ax = plt.subplots(figsize=(16, 6), dpi=300)
labels = np.arange(0, 200) #[] #[str(round(i, 0)) for i in sorted(g_agents[exampleNumber])]
width = 0.3

def make_frame(t):
     
    # clear
    ax.clear()
    #axcb.clear()
    
    t = int(t*3)
    #t = 1/fps
    
    #print(t)
    # plotting line
    ax.axhline(y=rich_limit, color='k', alpha=.2, lw=0.7, label='Initial total wealth') #richness line
    
    ax.bar(labels, Abonds[exampleNumber, sorted_idx, t], width, color='k', label='Av. Margin ($)')
    ax.bar(labels, Aequity[exampleNumber, sorted_idx, t], width, bottom=Abonds[exampleNumber, sorted_idx, t], color='r', label='Stock value owned ($)')
    
    #ax.colorbar(sc)
    #fig.colorbar(sc, ax=axcb)
    
    #fig.colorbar(sc, ax=ax, cax=axcb)
    #fig.subplots_adjust(wspace=0.1, hspace=0.5)
    ax.set_ylim([0, 30000])
    ax.set_xlabel(r'$g$')
    ax.set_ylabel('Total wealth ($)')

    ax.legend(loc=1)
    ax.set_title(f'Time step: {t}')
    plt.tight_layout()
    
    #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    #ax.color 
    
    # returning numpy image
    return mplfig_to_npimage(fig)
 
   
# creating animation
animation = VideoClip(make_frame, duration = duration)

#animation.write_gif('matplotlib.gif', fps=20)

animation.write_videofile(f"bars_exN_{exampleNumber}.mp4", fps=fps) 
