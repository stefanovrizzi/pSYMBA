#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:24:48 2022

@author: svrizzi
"""

################ Import key functions ############################

import sys
sys.path.append('..') if '..' not in sys.path else None  #go to parent folder
from plot.pretraining import PlotPretraining
from plot.training import PlotTraining
from plot.testing import PlotTesting
from plot.report import Reporting
#from plot.learning import PlotLearning

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching

import argparse

class SimulationAnalysis:
    
    def __init__(self, fetch):
    
        self.fetch = fetch
        
    def report(self):
    
        print('Reporting')
        
        self.plot = PlotTraining(self.fetch)
        reporting = Reporting(self.plot)
        #reporting.write_report()
        reporting.powerpoint()

    def plot_pretraining(self):
        
        print('Plot from pretraining simulations')

        self.plot = PlotPretraining(self.fetch)

        self.micro()
        
        del self.plot
        
        print('Done')
        
    def plot_training(self):
        
        print('Plot from training simulations')

        self.plot = PlotTraining(self.fetch)
         
        #self.plot.inflation_t0()
        #self.plot.inflation_t()
        #self.plot.trader_parameters_plot()   
        #self.plot.evolutionary_trading()
        #self.macro()
        #self.micro()
        #self.plot.fundamental_value_convergence() 
        self.micro_learning()
        
        #self.plot.forecast_preference()
        
        del self.plot
        
        print('Done')
        
    def macro(self):
        
        self.plot.macro_convergence()
        
        self.plot.market_dynamics()
        self.plot.return_correlation()
        #plotTraining.create_giff('trading')
        ##plotTraining.create_giff('QF')
        ##plotTraining.create_giff('QT')
        ##plotTraining.create_giff('wealth')
        ##plotTraining.convergence()
        #self.plot.trader_parameters_plot()
        #plotTraining.SYMBA_parameter_space()
        #self.plot.SYMBA_parameter_space_html()
        self.plot.price_volume_relationship()
        self.plot.returns_volume_change_relationship()
        #self.plot.zero_returns_trading_relationship()
    
    def micro(self):
    
        self.plot.attempts_vs_wealth()
        self.plot.trading_attempts()
        self.plot.success_parameter()
        self.plot.resolved_trading_parameters()
    
    def micro_learning(self):

        #self.plot.QT_visiting()
        #self.plot.QT_prob_irrational()
        
        #self.plot.forecasting_transition_matrix()
        #self.plot.QF_convergence_learning()
        #self.plot.QT_convergence_learning()
        self.plot.QP_convergence_learning()
        
        #self.plot.forecasting_state_visit()
        
        #self.plot.trading_state_visit()
        #self.plot.trading_transition_matrix()
        
        #self.plot.forecast_preference() 
        #self.plot.QF_distribution()        
        #self.plot.QT_distribution()
        #self.plot.QT_prob_distribution()
        #self.plot.QP_distribution()
        #self.plot.QP_prob_distribution()
        
        #self.plot.QF_dynamics()
        #self.plot.QT_dynamics()
        #self.plot.QP_dynamics()
        
    def plot_training_giffs(self):

        plotTraining = PlotTraining(self.fetch)
        #plotTraining.create_giff('trading')
        plotTraining.create_giff('QF')
        ##plotTraining.create_giff('QT')
        ##plotTraining.create_giff('wealth')
        
    def plot_training_videos(self):
    
        plotTraining = PlotTraining(self.fetch)
    
        #Videos
        #plotTraining.wealth_video()
        #plotTraining.pareto()
        #plotTraining.scatter_wealth_video()
        #plotTraining.QT_space_video()
        plotTraining.QT_radar_video()
        
        print('Done')
        
   
    def plot_testing(self):

        plotTesting = PlotTesting(self.fetch)
       
        plotTesting.AC_clustering()
       
        #plotTesting.price_buffer_returns2() #LSE Stock
        plotTesting.price_dynamics_fundamental_value()
        plotTesting.price_dynamics()
        plotTesting.forecast_preference()
        #plotTesting.QF_convergence_learning()
        #plotTesting.QF_convergence_learning_mean()
        #plotTesting.forecasting_transition_matrix()
          
        plotTesting.mean_square_displacement()
        #plotTesting.market_dynamics()
        plotTesting.SF_returns()
        plotTesting.autocorrelation()
        plotTesting.returns_distributions()
        ##plotTesting.parameter_analysis()
        ##plotTesting.traders_dynamics()
        plotTesting.stylised_facts()
        plotTesting.AC()
        plotTesting.AC_volatility()
        
        print('Done')
        
    def plot_learning(self):
    
        plotLearning = PlotLearning(self.fetch)
        #plotLearning.QF_dynamics()
        
def get_parser():
    
    #https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--exampleNumber', nargs='?', const=True, type=int, help = 'please give an integer number identifying an example already run', required=False, default=-1)
    parser.add_argument('--training', nargs='?', const=True, type=bool, help = '', required=False, default=True)
    parser.add_argument('--testing', nargs='?', const=True, type=bool, help = '', required=False, default=True)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    print('Check resolved trading vs parameters plot to evalute interactions of traders according to their tau!')
    
    args = get_parser()
        
    store = FileStorage()
    store.set_example(args.exampleNumber)
    fetch = FileFetching(store)

    simulationAnalysis = SimulationAnalysis(fetch)
    #simulationAnalysis.plot_pretraining()
    simulationAnalysis.plot_training() if args.training else None
    simulationAnalysis.plot_testing()  if args.testing else None
    ##simulationAnalysis.plot_learning()
    
    #simulationAnalysis.report()
    
    #simulationAnalysis.plot_training_videos()
    #simulationAnalysis.plot_training_giffs()
