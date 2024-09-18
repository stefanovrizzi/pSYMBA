#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 19:01:14 2023

@author: svrizzi
"""

################ Import key functions ############################

import os
from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
import statsmodels.api as sm
import pingouin as pg
from scipy.stats import ttest_rel
from scipy.stats import linregress
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
#import umap
from sklearn.manifold import TSNE
import argparse

from scipy.stats import wilcoxon

from matplotlib.ticker import ScalarFormatter

from plot.palette_colorblind import palette, conditions_order, dict_string2num, dict_num2label

import matplotlib as mpl
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
import string

sns.set_palette(palette)

class CompareAnalysis:
    
    def __init__(self, n):
                        
        self.general_setup()
        self.comparison_setup(n)
        
        ### Plots
        self.plot_macro()
        #self.plot_macro_bubbles()
        
        self.plot_MSD()
        
        ### Stats
        self.statistical_tests_macro('macro', r'$f$')
        self.statistical_tests_macro('macro', r'$Z$')
        self.statistical_tests_macro('macro', r'$B_{00}$')
        ##self.statistical_tests_macro('macro', 'LSE')
        
        self.statistical_tests_macro('bubble', r'$B_{00}$')
                
        #self.plot_micro()

    def general_setup(self):
    
        self.palette = palette
        self.order = conditions_order
        self.dict_string2num = dict_string2num
        self.dict_num2label = dict_num2label

        self.hurst = 'Hurst_adj' #'Hurst' #, 'Hurst_adj'
        self.hurst_col_name = r'$\mathcal{H}$' #if self.hurst == 'Hurst' else r'$\hat{\mathcal{H}}$' if self.hurst == 'Hurst_adj' else None


    def comparison_setup(self, n):
    
        self.store = FileStorage()
        fetch = FileFetching(self.store)
        comparison, n = fetch.comparison(n)

        self.ComparisonFolder = fetch.plotFolderPath+f'/comparison_{n}'
        os.makedirs(self.ComparisonFolder) if not os.path.isdir(self.ComparisonFolder) else None #create Data folder for all data files, in case it does not exist
        
        self.ComparisonFolder_tables = self.ComparisonFolder+f'/Tables'
        os.makedirs(self.ComparisonFolder_tables) if not os.path.isdir(self.ComparisonFolder_tables) else None #create Data folder for tables, in case it does not exist

        self.ComparisonFolder_figures = self.ComparisonFolder+f'/Figures'
        os.makedirs(self.ComparisonFolder_figures) if not os.path.isdir(self.ComparisonFolder_figures) else None #create Data folder for tables, in case it does not exist
        
        self.Tw, self.Tm, self.Ty = fetch.time_references()
        
        CognitiveTraits = comparison['cognitiveTraits']        
        self.randomSeeds = comparison['randomSeeds']
        exStart = comparison['exStart']
            
        C = len(CognitiveTraits)
        R = len(self.randomSeeds)
        
        self.datasets = range(exStart, exStart+R*C)
        self.R = R
        self.C = C
        
        self.store.set_example(self.datasets[0])
        self.fetch = FileFetching(self.store) 
        self.input_parameters()
        
        self.lags = np.logspace(4, 38, base=1.2).astype(int) #int(self.T/2))
        #self.lags = np.logspace(4, 26, base=1.2).astype(int) #int(self.T/2))        
        self.lags = list( set(self.lags) )
        self.lags.sort()
        #print(self.lags)
        self.sliding_window_view = np.lib.stride_tricks.sliding_window_view
     
    def load_datasets(self, dataframe_type=None):
            
        #loop over datasets
        for dataset in self.datasets:
            
            #self.dataset = dataset #for diagnosis
            
            self.store.set_example(dataset)
            
            try: #in case you delete folders
                self.fetch = FileFetching(self.store)  

                cognitivetraits = self.fetch.input_parameters()['CognitiveTraits'][0]
                self.bias = self.dict_string2num[cognitivetraits]
            
                #self.bias = 1 if 'optimistic bias' in cognitivetraits else 2 if 'no learning' in cognitivetraits else 0
            
                randomSeed = self.fetch.input_parameters()['randomSeed']
                self.rseed = self.randomSeeds.index(randomSeed)
            
                if dataframe_type in ['macro', 'MSD', 'Hurst', 'Hurst_adj', 'bubble']:
                    self.load_macrovariables(dataframe_type)
                elif variable=='micro':
                    self.load_microvariables()
                elif variable=='Q':
                    self.load_microvariables(Q=True)
                    
            except:
                pass
                
        #color = ['b', 'r']
        #plt.figure()
        #[plt.plot(self.Price[bias, rseed, s], label=f'{self.bias}', color=color[bias], alpha=.1) for s in range(self.S) for rseed, _ in enumerate(self.randomSeeds) for bias in [0, 1]]
            
        #plt.show()
            
        #self.LSE_data()

    def prepare_macro_df(self, dataframe_type):

        filepath = self.ComparisonFolder+f'/df_{dataframe_type}.csv'
        
        index = ["condition", "seed-sim"]

        if os.path.isfile(filepath):
        
            df = pd.read_csv(filepath)
            df.set_index(index, inplace=True)
        
        elif not os.path.isfile(filepath):

            self.initialise_macrovariables()
            self.load_datasets(dataframe_type=dataframe_type)

            if dataframe_type == 'macro':
            
                data = self.concat_adapt_data(self.dfmacro_list, index)
        
                datagrouped = data.groupby(index)
        
                df = pd.DataFrame()
                df[r'$\mu$'] = datagrouped.mean()
                #df[r'$\sigma^2$'] = datagrouped.var()
                df[r'$\log{\sigma^2}$'] = np.log( datagrouped.var() )
                
                df[r'$Skew$'] = datagrouped.skew()
                df[r'$Kurt$'] = datagrouped.apply(pd.DataFrame.kurt)
        
                df[r'$AC_{d}$'] = datagrouped.apply(lambda x : sm.tsa.acf(x, nlags=1)[-1])
                df[r'$AC_{m}$'] = datagrouped.apply(lambda x : sm.tsa.acf(x, nlags=self.Tm)[-1])
                df[r'$AC_{Y}$'] = datagrouped.apply(lambda x : sm.tsa.acf(x, nlags=self.Ty)[-1])
            
            elif dataframe_type == 'MSD':
            
                df = self.concat_adapt_data(self.dfMSD_list, index)
              
            elif dataframe_type == 'Hurst':
            
                df = self.concat_adapt_data(self.dfHurst_list, index)

            elif dataframe_type == 'Hurst_adj':

                df = self.concat_adapt_data(self.dfHurst_adj_list, index)

            elif dataframe_type == 'volatility':

                df = self.concat_adapt_data(self.volatility_list, index)
                
            elif dataframe_type == 'bubble':

                df = self.concat_adapt_data(self.dfBubble_list, index)
                    
            df.reset_index().to_csv(filepath, index=False)
        
        return df
    
    def concat_adapt_data(self, data_list, index):
    
        data = pd.concat(data_list)

        data['seed-sim'] = data['seed'].astype(str)+'-'+data['sim'].astype(str)
        data = data.drop(['seed', 'sim'], axis=1)    
       
        data.set_index(index, inplace=True)                
        #datagrouped = data.groupby(["condition", "seed-sim"])

        return data
    
    def MSD(self, variable):

    	# Mean Squared Displacement (MSD)

        SquaredPriceResiduals = np.zeros( len(self.lags) )

        for tn, t_ in enumerate(self.lags):
        
            PriceWindows = self.sliding_window_view(variable, window_shape = t_)
            price_residuals = lambda x: (np.log(x[-1]) - np.log(x[0]))**2
            SquaredPriceResiduals[tn] = np.array([price_residuals(x) for x in PriceWindows]).mean()

        return SquaredPriceResiduals # Raw MSD
        
    def Hurst(self, SquaredPriceResiduals):
        
        x = np.log( np.array(self.lags)-1 ) #lags are actually window sizes, so we need to count the actual time lags
        y = np.log( SquaredPriceResiduals )
        
        result = linregress(x, y)
        slope = result.slope
        
        return slope / 2 #Hurst exponent

    def Hurst_adj(self, price):
    #Eq. 10 in https://www.researchgate.net/profile/Matthieu-Garcin/publication/351699171_Forecasting_with_fractional_Brownian_motion_a_financial_perspective/links/61320d0f2b40ec7d8be02e67/Forecasting-with-fractional-Brownian-motion-a-financial-perspective.pdf
        
        T = len(price)
        #tau1, tau2 = 10, 100
        tau1, tau2 = 5, 50

        price_residuals = lambda x: (np.log(x[-1]) - np.log(x[0]))**2 
                
        PriceWindows = self.sliding_window_view(price, window_shape = tau1)        
        num = (T-tau2)*np.array([price_residuals(x) for x in PriceWindows]).sum()

        PriceWindows = self.sliding_window_view(price, window_shape = tau2)
        den = (T-tau1)*np.array([price_residuals(x) for x in PriceWindows]).sum()
        
        H_hat = 1/(2*np.log(tau1/tau2)) * np.log(num/den)
        
        return H_hat #Hurst exponent

    def add_LSE(self):

        filepath = self.ComparisonFolder+'/df_LSE.csv'
        
        index_name = 'condition'
        
        if os.path.isfile(filepath):
        
            df = pd.read_csv(filepath)
            df.set_index(index_name, inplace=True)
            
        elif not os.path.isfile(filepath):

            LSE = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)

            stockSelection = (LSE.diff(periods=1, axis=0).iloc[1:] == 0).mean() < .5 #active at least more than 50% of times
            stockSelection = pd.DataFrame(stockSelection, columns=['Active'])
            stockSelection = stockSelection[stockSelection['Active']==True].index
            self.LSE = LSE[stockSelection]
        
            LSEreturns = np.log(LSE).diff(periods=1, axis=0).iloc[1:] #take log-returns

            df = pd.DataFrame()
            df[r'$\mu$'] = LSEreturns.mean()
            #df[r'$\sigma^2$'] = LSEreturns.var()
            df[r'$\log{\sigma^2}$'] = np.log( LSEreturns.var() )
            df[r'$Skew$'] = LSEreturns.skew()
            df[r'$Kurt$'] = pd.DataFrame.kurt(LSEreturns)
            df[r'$AC_{d}$'] = LSEreturns.apply(lambda x : sm.tsa.acf(x, nlags=1)[-1])
            df[r'$AC_{m}$'] = LSEreturns.apply(lambda x : sm.tsa.acf(x, nlags=self.Tm)[-1])
            df[r'$AC_{Y}$'] = LSEreturns.apply(lambda x : sm.tsa.acf(x, nlags=self.Ty)[-1])
            
            df[self.hurst_col_name] = LSE.apply(lambda x: self.Hurst( self.MSD(x) ) ) if self.hurst == 'Hurst' else LSE.apply(lambda x: self.Hurst_adj( x ) ) if self.hurst == 'Hurst_adj' else None

            df.index = ['LSE']*len(df.index) #index must be 'LSE' for all entries (stocks)
            df.index.name = index_name
            df.reset_index().to_csv(filepath, index=False)

        return df
        
    def find_lim(self, df, column):

        xmin = df[ column ].min().min()
        xmax = df[ column ].max().max()   
        xmin = xmin*1.25*(xmin < 0)+xmin*0.75*(xmin > 0)
        xmax = xmax*1.25*(xmax > 0)+xmax*0.75*(xmax < 0)
        
        return xmin, xmax

    def plot_macro(self):
    
        df = self.prepare_macro_df('macro')
        dfHurst = self.prepare_macro_df(self.hurst)
        df = df.join(dfHurst)

        AClist = [r'$AC_{d}$', r'$AC_{m}$', r'$AC_{Y}$']
        #timeIndependentStats = [r'$\mu$', r'$\sigma^2$', r'$Skew$', r'$Kurt$']
        timeIndependentStats = [r'$\mu$', r'$\log{\sigma^2}$', r'$Skew$', r'$Kurt$']
        
        # Autocorrelation ax limits to check decay
        mumin, mumax = self.find_lim(df, df.columns[0])
        sigmamin, sigmamax = self.find_lim(df, df.columns[1])
        skewmin, skewmax = self.find_lim(df, df.columns[2])
        kurtmin, kurtmax = self.find_lim(df, df.columns[3])

        acmin, acmax = self.find_lim(df, AClist)
        
        LSE = self.add_LSE()

        df = df.droplevel('seed-sim')
        
        x = 'condition'
                
        df = pd.concat([df, LSE])
        df = df.reset_index()

        df[x] = pd.Categorical(df[x], categories=self.order, ordered=True)
        df = df.sort_values(by='condition')        
        
        #print(df.index.get_level_values( x ).unique() ) #to find palette order

        palettemeanstrip = ['w',      #CB
                            'w',      #PB
                            'w',      #NNB
                            'w',      #NB                            
                            'w',      #NL
                            'k',      #EI
                            'w']      #LSE
                            
        Nconditions = len( df[x].unique() )
        variables = df.columns[1:]
        
        palettemeanstrip = palettemeanstrip[:Nconditions]
        
        _, axs = plt.subplots(2, np.ceil( len(variables)/2 ).astype(int), figsize=(10, 6), sharex=True, sharey=False)
        axs = axs.flatten()

        df_mean = df.groupby(x, observed=False).mean()
        df_mean.to_csv(self.ComparisonFolder+'/df_means.csv')
        
        df_median = df.groupby(x, observed=False).median()
        df_median.to_csv(self.ComparisonFolder+'/df_medians.csv')
        
        #df_mean[x] = pd.Categorical(df_mean[x], categories=self.order, ordered=True)
        #df_median[x] = pd.Categorical(df_median[x], categories=self.order, ordered=True)
        
        for columnN, column in enumerate(variables): #exclude 'condition'
            
            ax = axs[columnN]
            
            yline = .5 if column == self.hurst_col_name else 0
            ax.axhline(y=yline, color=[.8, .8, .8], linewidth=.8, alpha=.5)
            
            sns.violinplot(data=df, y=column, x=x, inner=None, ax=ax, hue=x, palette=palette, legend=False)
            
            for i, violin in enumerate(ax.collections[:-1]): #no alpha on LSE
                violin.set_alpha(.4) #, violin.set_facecolor(paletteviolin[i])
                #violin.set_edgecolor('k')

            stripplot = sns.stripplot(x=x, y=column, data=df, palette=self.palette[:-1]+['#ececec'], hue=x, size=2, jitter=True, ax=ax)            
            sns.stripplot(df_median, x=x, y=column, palette=['w']*len(df_median), hue=x, color='k', edgecolor='k', linewidth=.5, size=5, legend=False, marker="o", ax=ax)

            #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            
            # Set the power limits to use scientific notation only for large ranges
            ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
            
            #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
                                
            #p = p.round(3)           
            #significant = '*' if p<threshold else ''
            ##ax.set_title(fr'$p^{significant}={p}$')
            ##pass
            ##ax.text(0.85, 0.85, fr'$p^{significant}={p}$', fontsize=9) #add text
            ax.text(-0.3, 1.065, string.ascii_lowercase[columnN], transform=ax.transAxes, size=12, weight='bold')
        
            if column in AClist:
                ax.set_ylim([acmin, acmax])
            elif columnN == 0:
                ax.set_ylim([mumin, mumax])
            elif columnN == 1:
                ax.set_ylim([sigmamin, sigmamax])
                #ax.set_ylim([-11, sigmamax])                
                labels = [label.get_text() for label in ax.get_yticklabels()]
                #ax.set_yticklabels(map(lambda x: "{:g}".format(float(x)), labels))
                #ax.set_yticklabels(map(lambda x: "{:g}".format(float(x.replace('−', '-'))), labels))
                #ax.set_yticklabels([f"{x:.1f}" for x in ax.get_yticks()])

            elif columnN == 2:
                ax.set_ylim([skewmin, skewmax])
            elif columnN == 3:
                ax.set_ylim([kurtmin, kurtmax])
        
        plt.tight_layout()
        plt.savefig(self.ComparisonFolder_figures+f'/macro_returns_{self.hurst}.pdf', dpi=300)
        plt.close()

    def plot_macro_bubbles(self):
    
        df = self.prepare_macro_df('bubble')     

        df = df.droplevel('seed-sim')

        #df[r'$\log_{10}{ \text{MSI} }$'] = np.log10(df[r'$\Delta_f$'])
        
        x = 'condition'  
        
        Nconditions = 5
        
        df = df.reset_index()
   
        df[x] = pd.Categorical(df[x], categories=self.order[:Nconditions], ordered=True)
        df = df.sort_values(by='condition')
                        
        fig, axs = plt.subplots(1, 4, figsize=(8, 4))
        axs = axs.flatten()
        
        palette = self.palette[:Nconditions]

        df_mean = df.groupby(x, observed=False).mean()
        df_mean.to_csv(self.ComparisonFolder+'/df_means_bubbles.csv')
        df_mean.reset_index(inplace=True)
        df_mean[x] = pd.Categorical(df_mean[x], categories=self.order[:Nconditions], ordered=True)        
        df_mean = df_mean.sort_values(by='condition')

        
        df_median = df.groupby(x, observed=False).median()
        df_median.to_csv(self.ComparisonFolder+'/df_medians_bubbles.csv')
        df_median.reset_index(inplace=True)
        df_median[x] = pd.Categorical(df_median[x], categories=self.order[:Nconditions], ordered=True)        
        df_median = df_median.sort_values(by='condition')

        for columnN, column in enumerate(df.columns[1:]): #exclude 'condition'
                         
            ax = axs[columnN]
        
            sns.violinplot(data=df, x=x, y=column, palette=palette, ax=ax, hue=x, split=False, inner=None, legend=False)

            for i, violin in enumerate(ax.collections):
                violin.set_alpha(.4)

            sns.stripplot(data=df, y=column, x=x, hue=x, palette=palette, s=1, ax=ax, legend=False)
            sns.stripplot(df_median, x=x, y=column, palette=['w', 'w', 'w', 'w', 'w'], hue=x, color='k', edgecolor='k', linewidth=.5, size=5, legend=False, marker="o", ax=ax)

            ax.set_ylim([0, None]) if columnN in [1, 2] else None            
            #ax.set_yscale('log') if col_n in [1, 2] else None
            
            ax.text(-0.4, 1, string.ascii_lowercase[columnN], transform=ax.transAxes, size=12, weight='bold')
                    
        plt.tight_layout()
        plt.savefig(self.ComparisonFolder_figures+'/macro_bubbles_violin.pdf', dpi=300)
        plt.close()

        
    def statistical_tests_macro(self, df_name, baseline_condition):

        df = self.prepare_macro_df(df_name)

        if df_name == 'macro':

            dfHurst = self.prepare_macro_df(self.hurst)
            df = df.join(dfHurst)

        df_columns = df.columns.tolist()

        if baseline_condition == r'$Z$':
            df = df.drop(r'$f$', axis=0)
        elif baseline_condition in [r'$B_{00}$', r'$B_{+}$']:
            df = df.drop([r'$f$', r'$Z$'], axis=0, level=0) if df_name == 'macro' else df.drop(r'$Z$', axis=0, level=0) if df_name == 'bubble' else None

        # Step 1: Calculate median of differences
        df_change = (df.drop(baseline_condition)-df.xs(baseline_condition, level='condition')).groupby('condition').median()
                    
        #df_change = ( frac ) *100
        #df_change = df_change.round(2)
        df_change = df_change.reset_index()
        df_change['condition'] = pd.Categorical(df_change['condition'], categories=self.order, ordered=True)
        df_change = df_change.set_index('condition')
        df_change = df_change.sort_values(by='condition')

        # Sort the DataFrame by the categorical index
        #df_change = df_change.sort_index(level='condition')

        # Dictionary to store p-values for pairwise comparisons
        p_values = {}
        stat_values = {}

        self.n_corrections = len( df_change.index )

        for column in df_change.columns:

            p_values[column] = {}
            stat_values[column] = {}

            for condition in df_change.index:
                
                stat_values[column][condition], p_values[column][condition] = self.wilcoxon_test(df.xs(condition)[column], df.xs(baseline_condition)[column])

        cutoff_threshold = 0.05 / len(df_change.index) #len(df_change.columns)  #* len(df_change.index) )

        # Create LaTeX tables for each analysis
        latex_table_analysis = self.create_latex_table(df_change, df_name, p_values, cutoff_threshold, stat_values, baseline_condition)

    # Step 2: Perform Pairwise Comparisons and Calculate p-values    
    # Function to perform pairwise Wilcoxon Signed-Rank Test with Bonferroni correction
    
    def wilcoxon_test(self, series1, series2):
        stat_val, p_val = wilcoxon(series1, series2)
        return stat_val, p_val

    def format_value(self, value):
        # Format the value in scientific notation
        formatted_p_value = f"{value:.2e}"
    
        # Split into base and exponent
        base, exponent = formatted_p_value.split('e')
    
        # Format exponent to remove leading zero
        exponent = exponent.replace('+', '')  # Remove leading plus sign
        if exponent.startswith('0'):
            exponent = exponent[1:]  # Remove leading zero for positive exponents
        elif exponent.startswith('-0'):
            exponent = exponent[0]+exponent[2:]

        # Construct final formatted string
        return f"{base} \\times 10^{{{exponent}}}"

    # Step 3: Create LaTeX Tables

    def create_latex_table2(self, df, df_name, p_values, cutoff_threshold, stat_values, baseline_condition):

        # Begin the table environment
        latex_table = (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\begin{tabular}{|c|" + "c|" * len(df.columns) + "}\n\\hline\n"
        )
        
        # Header row for variables
        latex_table += "& " + " & ".join([f"\\textbf{{{var}}}" for var in df.columns]) + " \\\\ \n\\hline\n"

        # Iterate over rows (conditions)
        for condition in df.index:
            # First row: % change values
            row_content = []
            for var in df.columns:
                change = df.loc[condition, var]
                change_str = f"{change:.2f}" if abs(change) >= 0.01 else self.format_value(change)
                change_str = f"${change_str}$"
                
                p_value = p_values[var][condition]
                p_value_str = f"{p_value:.2f}" if p_value >= 0.01 else self.format_value(p_value)
                p_value_str = f"p={p_value_str}"
                p_value_str = f"${p_value_str}$"

                if p_value < cutoff_threshold and change > 0:
                    color = "lightforestgreen"
                elif p_value < cutoff_threshold and change < 0:
                    color = "salmonpink"
                else:
                    color = "white"  # Default color
            
                stat_value = stat_values[var][condition]
                stat_value_str = f"{stat_value:.2f}" if stat_value >= 0.01 else self.format_value(stat_value)
                stat_value_str = f"W={stat_value_str}"
                stat_value_str = f"${stat_value_str}$"

                # Create the multi-line cell content
                cell_content = f"{change_str}\\\\ {p_value_str}\\\\ stat={stat_value_str}"
                
                # Construct the LaTeX code for the cell
                cell_code = (
                    f"\\cellcolor{{{color}}}"
                    f"\\begin{{tabular}}{{@{{}}c@{{}}}}"
                    f"{cell_content}"
                    f"\\end{{tabular}}"
                )
                
                row_content.append(cell_code)

        
            latex_table += f"\\textbf{{{condition}}} & " + " & ".join(row_content) + " \\\\ \n\\hline\n"
        
        # End of the table environment
        latex_table += "\\end{tabular}\n\\caption{Your caption here. Comparing "
        latex_table += f"{baseline_condition} to other conditions.}}"
        latex_table += "\n\\label{yourlabel}\n\\end{table}"

        # Define the file path using the baseline_condition variable
        file_path = self.ComparisonFolder_tables + f'/table_output_{baseline_condition}_{df_name}.tex'

        # Write the LaTeX table to a file
        with open(file_path, 'w') as f:
            f.write(latex_table)
        
        return latex_table

    def create_latex_table(self, df, df_name, p_values, cutoff_threshold, stat_values, baseline_condition):

        # Begin the table environment
        latex_table = (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\begin{tabular}{|c|" + "c|" * len(df.index) + "}\n\\hline\n"
        )
        
        # Header row for conditions (bold titles)
        latex_table += "& " + " & ".join([f"\\textbf{{{condition}}}" for condition in df.index]) + " \\\\ \n\\hline\n"
        
        # Iterate over columns (variables)
        for var in df.columns:
            # First row: % change values
            row_content = []
            for condition in df.index:
                change = df.loc[condition, var]
                change_str = f"{change:.2f}" if abs(change) >= 0.01 else self.format_value(change)
                change_str = f"${change_str}$"
                
                p_value = p_values[var][condition]
                p_value_str = f"{p_value:.2f}" if p_value >= 0.01 else self.format_value(p_value)
                p_value_str = f"p={p_value_str}"
                p_value_str = f"${p_value_str}$"

                if p_value < cutoff_threshold and change > 0:
                    color = "lightforestgreen"
                elif p_value < cutoff_threshold and change < 0:
                    color = "salmonpink"
                else:
                    color = "white"  # Default color
            
                stat_value = stat_values[var][condition]
                stat_value = int(stat_value)
                stat_value_str = f"{stat_value}" #f"{stat_value:.2f}" if stat_value >= 0.01 else self.format_value(stat_value)
                stat_value_str = f"W={stat_value_str}"
                stat_value_str = f"${stat_value_str}$"

                # Create the multi-line cell content with added vertical space
                cell_content = f"\\vspace{{0.2em}}{change_str}\\\\ {p_value_str}\\\\ {stat_value_str}"
                
                # Construct the LaTeX code for the cell
                cell_code = (
                    f"\\cellcolor{{{color}}}"
                    f"\\begin{{tabular}}{{@{{}}c@{{}}}}"
                    f"{cell_content}"
                    f"\\end{{tabular}}"
                )
                
                row_content.append(cell_code)

        
            # Add bold for the first column (variables)
            latex_table += f"\\textbf{{{var}}} & " + " & ".join(row_content) + " \\\\ \n\\hline\n"
        
        # End of the table environment
        latex_table += "\\end{tabular}\n\\caption{Your caption here. Comparing "
        latex_table += f"{baseline_condition} to other conditions.}}"
        latex_table += "\n\\label{yourlabel}\n\\end{table}"

        # Define the file path using the baseline_condition variable
        file_path = self.ComparisonFolder_tables + f'/table_output_{baseline_condition}_{df_name}.tex'

        # Write the LaTeX table to a file
        with open(file_path, 'w') as f:
            f.write(latex_table)
        
        return latex_table



    # Function to apply LaTeX formatting based on cell values
    def apply_latex_formatting(self, cell):
            
        cell_str = str(cell)
            
        if cell_str.startswith('-') and cell_str.endswith('*'):
            return f'\\cellcolor{{red!50}}{cell_str}'
        elif cell_str[0].isdigit() and cell_str.endswith('*'):
            return f'\\cellcolor{{green!50}}{cell_str}'
        else:
            return cell_str
   
    def latex_table_image(self, filepath, columnN, column):
    
        # Read the Excel file
        df = pd.read_excel(filepath, index_col=0)
        df.reset_index(inplace=True)
        df.rename(columns={'index':''}, inplace=True)
        df.replace(np.nan, ' ', inplace=True)
        
        # Apply the formatting function to the DataFrame
        latex_df = df.applymap(self.apply_latex_formatting)

        # Convert the DataFrame to a LaTeX tabular environment
        #latex_table = latex_df.to_latex(escape=False, index=False, index_names=False)
        
        # Manually construct the LaTeX table content
        header_row = ' & '.join(latex_df.columns)
        latex_table_content = '\n'.join([' & '.join(row) + ' \\\\' for row in latex_df.values])
        
        # Create the LaTeX document with adjusted dimensions
        latex_document = r"""
        \documentclass{article}
        \usepackage[table]{xcolor}
        \usepackage{colortbl}
        \usepackage{graphicx}
        \pagestyle{empty}
        \begin{document}
        \begin{minipage}[t]{\textwidth}
        \centering
        \textbf{%s}\\
        \vspace{0.5cm}
        \begin{tabular}{ccccc}
        \toprule
        %s \\
        \midrule
        %s
        \bottomrule
        \end{tabular}
        \end{minipage}
        \end{document}
        """ % (column, header_row, latex_table_content)

        # Construct the file name based on the variable value
        file_name = self.ComparisonFolder+f'/table_{columnN}.tex'

        # Save the LaTeX document to the file with the constructed name
        with open(file_name, 'w') as f:
            f.write(latex_document)

        # Compile the LaTeX document to a PDF
        os.system(f'pdflatex -interaction=batchmode {file_name}')

        # Crop the PDF to remove white space
        os.system(f'pdfcrop {file_name.replace(".tex", ".pdf")}')

        # Convert the cropped PDF to a PNG image with 300 DPI
        os.system(f'pdftoppm -png -r 300 {file_name.replace(".tex", "-crop.pdf")} {file_name.replace(".tex", "")}')

        ### MOVE FILES TO self.ComparisonFolder !
    
    def pairwise_test(self, df, column, x_name, y_name, dfp):
            
        x = df.xs(level='condition', key=x_name)[column]
        y = df.xs(level='condition', key=y_name)[column]
            
        res = stats.wilcoxon(x, y)
        p = res.pvalue
        print('W test:', column, x_name, y_name, p, 'significant:', p<self.threshold, 'delta mean', abs(x.mean()/y.mean()-1)*100*np.sign(x.mean()-y.mean()) )
        
        dfp_temp = pd.DataFrame()
        #dfp_temp['variable'] = [column]
        dfp_temp['x'] = [x_name]
        dfp_temp['y'] = [y_name]
        ast = p<self.threshold
        ast = '*' if ast else ''
        dfp_temp['% change'] = [str( round( abs(x.mean()/y.mean()-1)*100*np.sign(x.mean()-y.mean()), 2) ) +ast]

        dfp = pd.concat([dfp, dfp_temp])
        
        return dfp

    def plot_MSD(self):
        
        df = self.prepare_macro_df('MSD')
        df.reset_index(inplace=True)

        df['condition'] = pd.Categorical(df['condition'], categories=self.order[:5], ordered=True)
        df = df.sort_values(by='condition')
                                       
        sns.lineplot(data=df, x='lags', y=r'$MSD$', hue='condition', palette=self.palette[:5], alpha=.7)

        plt.xlabel('Time lag '+r'$(t-\tau)$')
        plt.ylabel('Mean '+r'$[r(t-\tau, t)^2]$')
        plt.xscale('log')
        plt.yscale('log')
        
        # Define the identity line y = x
        #min_val = df[r'$MSD$'].min()  # Get the minimum x value from your data
        min_val = df['lags'].min()  # Get the minimum x value from your data        
        max_val = df['lags'].max()  # Get the maximum x value from your data
        identity_line = np.linspace(min_val, max_val, 100)  # Create points for the identity line
        
        # Plot the identity line
        plt.plot(identity_line, identity_line, ls='--', color=[.5, .5, .5], linewidth=.5) #label='Identity line (y=x)',
        
        plt.legend()
        plt.savefig(self.ComparisonFolder_figures+f'/MSD.pdf', dpi=120)  

    def plot_micro(self):

        self.coefName = r'$\hat{\beta}$'

        self.initialise_microvariables()
        self.load_datasets(variable='micro')        
        data = pd.concat(self.dfmicro_list)
        
        #self.plot_micro_wealth(data)    
        #self.plot_micro_CB_tau(data)
        self.plot_micro_umap(data)
        #self.plot_micro_wealth_identity(data)

        #self.load_datasets(variable='Q')
        #dataQT = pd.concat(self.dfQT_list)
        #self.plot_micro_QT_prob(dataQT)
        #self.plot_micro_wealth_coef()
    
    def test_norm(self, x):
        normalized_x=(x-x.mean())/x.std()
        return stats.kstest(normalized_x, stats.norm().cdf )

    def test_unif(self, x):
        normalized_x=(x-x.min())/(x.max()-x.min())
        return stats.kstest(normalized_x, stats.uniform().cdf )

    def test_regressors_distributions(self, data, regressionVariables):
    
        dfp = pd.DataFrame(index=regressionVariables)
        dfp['Norm'] = data[regressionVariables].apply(lambda x: self.test_norm(x) ).T[1]
        dfp['Unif'] = data[regressionVariables].apply(lambda x: self.test_unif(x) ).T[1]
        
        return dfp

    def plot_regressors_distributions(self, data, regressionVariables, transformed=False):
        
        bias = data.bias.unique()
        
        fig = data[regressionVariables].hist(figsize=(15, 12))
        [x.title.set_size(12) for x in fig.ravel()]
        plt.tight_layout()
        title = f'/histplot_coeff_{bias}'
        title += '_transformed' if transformed else ''
        title += '.pdf'
        plt.savefig(self.ComparisonFolder+title)
        plt.close()

    def merge_micro_dataframes(self, data):
    
        data = data.reset_index()
        data = data.drop( data[data[r'$a_T$']=='Hold'].index ) #to avoid multicollinearity
        
        data = data.set_index(["bias", "seed", "sim", "ID", r'$s_0$', r'$s_1$', r'$a_T$']).unstack([r'$s_0$', r'$s_1$', r'$a_T$'])[self.QTName]
        data.columns = ["-".join(col) for col in data.columns.to_flat_index()]
        self.Qpredictors = data.columns.tolist()
                
        self.load_datasets(variable='micro')        
        dataW = pd.concat(self.dfmicro_list)
        dataW = dataW.set_index(data.index.names)
        
        df = data.merge(dataW, left_on=data.index.names, right_index=True).reset_index('condition')

        return df

    def set_df_micro_merged(self):

        self.load_datasets(variable='Q')
        data = pd.concat(self.dfQT_list)
        
        data = self.merge_micro_dataframes(data)

        self.store.df_micro_merged(data.reset_index(), self.ComparisonFolder)
        
        return data

    def transform_regressors(self, data, variablesToTransform):
    
        pt = QuantileTransformer() #PowerTransformer(method='yeo-johnson')
        pt.fit(data[variablesToTransform])
        transformed_data = pt.transform(data[variablesToTransform])
        data[variablesToTransform] = transformed_data

        return data

    def coef_regressors(self, data, predictors):
    
        X = data[predictors].to_numpy()
        y = data[self.WName].to_numpy()
        reg = LinearRegression().fit(X, y)
        coef = pd.DataFrame(reg.coef_, index=predictors, columns=[self.coefName])
        coef["bias"] = data.bias.unique()[0]

        return coef

    def set_df_micro_coef(self):
        
        data = self.set_df_micro_merged()
        
        predictorsNoBias = self.parametersNamesNoBias+self.Qpredictors
        regressionVariablesNoBias = predictorsNoBias+[self.WName]

        predictorsBias = self.parametersNames+self.Qpredictors
        regressionVariablesBias = predictorsBias+[self.WName]
        
        keys = ['parametersNames', 'parametersNamesNoBias', 'Qpredictors', 'predictorsBias', 'predictorsNoBias', 'regressionVariablesBias', 'regressionVariablesNoBias']
        values = [self.parametersNames, self.parametersNamesNoBias, self.Qpredictors, predictorsBias, predictorsNoBias, regressionVariablesBias, regressionVariablesNoBias]
        self.regressorDict = dict(zip(keys, values))
        self.store.regressor_dict(self.regressorDict, self.ComparisonFolder)
            
        dataNoBias = data[data.bias==0]
        dataBias = data[data.bias==1]

        self.plot_regressors_distributions(dataNoBias, regressionVariablesNoBias)
        self.plot_regressors_distributions(dataBias, regressionVariablesBias)

        ### Test to check normality or uniform distribution ###
                
        dfpNoBias = self.test_regressors_distributions(dataNoBias, regressionVariablesNoBias)
        dfpBias = self.test_regressors_distributions(dataBias, regressionVariablesBias)

        thresholdNoBias = 0.05 / len(regressionVariablesNoBias)
        thresholdBias = 0.05 / len(regressionVariablesBias)
        
        dfpNoBias['transform'] = (dfpNoBias > thresholdNoBias).sum(axis=1) <1
        dfpBias['transform'] = (dfpBias > thresholdBias).sum(axis=1) <1
                
        variablesToTransformNoBias = regressionVariablesNoBias #dfpNoBias[dfpNoBias['transform']==True].index.tolist() 
        variablesToTransformBias = regressionVariablesBias #dfpBias[dfpBias['transform']==True].index.tolist()

        ### Transform ###
                
        dataNoBias = self.transform_regressors(dataNoBias, variablesToTransformNoBias)
        dataBias = self.transform_regressors(dataBias, variablesToTransformBias)
        
        ### Check transformation ###

        self.plot_regressors_distributions(dataNoBias, regressionVariablesNoBias, transformed=True)
        self.plot_regressors_distributions(dataBias, regressionVariablesBias, transformed=True)
                        
        coefNoBias = self.coef_regressors(dataNoBias, predictorsNoBias)
        coefBias = self.coef_regressors(dataBias, predictorsBias)
                
        df = pd.concat([coefBias, coefNoBias])

        self.store.df_micro_coef(df.reset_index(), self.ComparisonFolder)
        
        return df


    def plot_micro_wealth_coef(self):
        
        df, self.regressorDict = self.fetch.df_micro_coef(self.ComparisonFolder)

        if df is None:
            df = self.set_df_micro_coef()
        else:
            self.parametersNamesNoBias = self.regressorDict['parametersNamesNoBias']
            self.Qpredictors = self.regressorDict['Qpredictors']
                    
        hue = 'condition'
        palette = [[.5, .5, .5], 'k']
        #paletteviolin = ['w', [.5, .5, .5]]
        
        threshold = 0.05 / len(df.columns)
        
        _, axs = plt.subplots(3,  1, figsize=(12, 8), sharex=False, sharey=True)
        axs = axs.flatten()
        
        x1 = self.parametersNamesNoBias
        x2 = [x for x in self.Qpredictors if x.startswith(r'$T$')]
        x3 = [x for x in self.Qpredictors if x.startswith(r'$F$')]
                
        xlist = [df.loc[x1], df.loc[x2], df.loc[x3]]
        
        for x, ax in zip(xlist, axs):      
        
            g = sns.barplot(data=x, y=self.coefName, x=x.index, hue=hue, palette=palette, ax=ax)
            #[ item.set_rotation(45) for item in g.get_xticklabels() ]
            g.set(xlabel='', xticks=g.get_xticks())
            #g.set_()
            ax.set_xticklabels(g.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
            ax.tick_params(axis='x', which='major', labelsize=9)
        
        plt.tight_layout()
        plt.savefig(self.ComparisonFolder+'/barplot_micro_coeff_to_wealth.pdf')
        plt.close()
        
        #test differences
        
        #for columnN, column in enumerate(df.columns):

            #Test
            #anova = df.reset_index().anova(dv=column, between=["bias"]).round(3)
            #p = anova['p-unc'].values[0]
            #significant = '*' if p<threshold else ''
            #ax.set_title(fr'$p^{significant}={p}$')
            #ax.text(0.85, 0.85, fr'$p^{significant}={p}$', fontsize=9) #add text

    def plot_micro_QT_prob(self, data):

        hue = r"$a_T$"
        palette= [[0, .5, 0], [.3, .3, .3], [.5, 0, 0]]
       
        data = data.reset_index().set_index(["bias", "seed", "sim", "ID", r"$s_0$", r"$s_1$", hue, r'$a_{order}$'])
        df = data.unstack(['condition'])[self.QTName]

        x = self.QTName+' (No bias)'
        y = self.QTName+' (Bias)'
        
        df = df.rename(columns={0:x, 1:y})
        df = df.reset_index(["seed", "sim", "ID", hue, r'$a_{order}$'])
        
        df.sort_values(by=[r'$a_{order}$'], ascending=[True], inplace=True)
        
        for idx in df.index.unique():
        
            (s0, s1) = idx
        
            datatemp = df.loc[idx]
        
            plt.figure(figsize=(8, 8))
            g = sns.lmplot(data=datatemp, x=x, y=y, hue=hue, palette=palette, scatter_kws={'alpha':0.05, 's':1}, lowess=False)
            plt.plot(datatemp[x], datatemp[x], color=[.8, .8, .8], linestyle='--', lw=1) # identity line
   
            for h in g._legend.legend_handles:
                h.set_alpha(1)
                h.set_sizes([20])
            
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            plt.savefig(self.ComparisonFolder+f'/micro_QT_prob_s0_{s0}_s1_{s1}.pdf')
            plt.close()

    def plot_micro_wealth(self, data):
    
        hue = 'condition'
        palette= [[0, 0, .5], [.5, 0, 0]]
        
        plt.figure(figsize=(8, 8))
        sns.lmplot(data=data, x=self.alphaDiscName, y=self.WName, hue=hue, palette=palette, scatter_kws={'alpha':0.25, 's':.4}) #, alpha=.5) #lowess=True,
        
        #lowess = sm.nonparametric.lowess
        #w = lowess(y, x, frac=1./3)
                
        plt.savefig(self.ComparisonFolder+'/micro_wealth.pdf')
        plt.close()
  
    def plot_micro_CB_tau(self, data):
    
        data['Confirmation bias'] = data[self.alphaConfName]-data[self.alphaDiscName]
        
        dftemp = data[data.bias==1]
        #dftemp = dftemp.groupby(['seed', 'ID']).mean()

        lims = max(-dftemp[self.WName].min(), dftemp[self.WName].max())
        
        plt.figure(figsize=(8, 8))
        
        #g = plt.scatter(dftemp['Confirmation bias'], dftemp[self.tauName], alpha=0.7, s=20, c=dftemp[self.WName], cmap='RdBu_r', vmin=-lims, vmax=lims)
        g = sns.lmplot(data=dftemp, x='Confirmation bias', y=self.WName, lowess=True, scatter_kws={'alpha':0.25, 's':.4, 'color':[.2, .2, .2]}, line_kws={'color':'k'})#, lowess=True
        plt.axhline(y=0, color=[.5, .5, .5], ls='--', lw=.5, alpha=.8)
        plt.axvline(x=0, color=[.5, .5, .5], ls='--', lw=.5, alpha=.8)
        #plt.colorbar(g)        
        plt.tight_layout()
                
        plt.savefig(self.ComparisonFolder+'/micro_CB.pdf')
        plt.close()  

    def plot_micro_umap(self, data):

        #data['Confirmation bias'] = data[self.alphaConfName]-data[self.alphaDiscName]

        data = self.fetch.df_micro_merged(self.ComparisonFolder)

        if data is None:
            data = self.set_df_micro_merged()
            
        _, self.regressorDict = self.fetch.df_micro_coef(self.ComparisonFolder)
        
        parametersNames = self.regressorDict['parametersNames']
        Qpredictors = self.regressorDict['Qpredictors']
        
        dftemp = data
        dftemp = data[data.bias==1]
        dftemp = dftemp.groupby(['seed', 'ID']).mean()
        
        #xfeatures = parametersNames+Qpredictors+['bias']
        xfeatures = parametersNames+Qpredictors
        
        X = dftemp[xfeatures].values
        y = dftemp[self.WName].values.flatten()
        
        pipe = make_pipeline(SimpleImputer(strategy='mean'), QuantileTransformer())
        X = pipe.fit_transform(X.copy())
            
        #Xreduced = umap.UMAP(n_neighbors=10).fit(X, y).transform(X)
        Xreduced = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50).fit_transform(X)

        lims = max(-dftemp[self.WName].min(), dftemp[self.WName].max())
        
        plt.figure(figsize=(8, 8))
        
        g = plt.scatter(Xreduced[:, 0], Xreduced[:, 1], alpha=0.7, s=10, c=dftemp[self.WName], cmap='RdBu_r', vmin=-lims, vmax=lims)
        plt.colorbar(g)        
        #plt.tight_layout()
                
        plt.savefig(self.ComparisonFolder+'/micro_tSNE.pdf')
        plt.close()  

    def plot_micro_wealth_identity(self, data):
    
        data.set_index(["condition", "seed", "sim", "ID"], inplace=True)
        datatemp = data.unstack(['condition'])[self.WName]

        x = self.WName+' (No bias)'
        y = self.WName+' (Bias)'
        
        datatemp = datatemp.rename(columns={0:x, 1:y})

        plt.figure(figsize=(8, 8))
        ax = sns.regplot(data=datatemp, x=x, y=y, color=[.5, .5, .5], scatter_kws={'alpha':0.1}, line_kws={'color':'k'}) #, alpha=.3)
        plt.axhline(y=0, color='k', alpha=.2)
        plt.axvline(x=0, color='k', alpha=.2)
        plt.plot(datatemp[x], datatemp[x], color='k', alpha=.2, linestyle='--', lw=.5) # identity line
        
        plt.savefig(self.ComparisonFolder+'/micro_wealth_identity.pdf')
        plt.close()
    
    def input_parameters(self):

        #Input parameters
        self.InputParameters = self.fetch.input_parameters()
        self.I = self.InputParameters['I']
        self.J = self.InputParameters['J']
        
        self.InputTimeParameters = self.fetch.input_parameters(time=True)
        self.T = self.InputTimeParameters['T']
        self.S = self.InputTimeParameters['S']
        
        SYMBAParameters = self.fetch.parameters_json('SYMBA')
        self.tInit = SYMBAParameters['tInit']

    def initialise_macrovariables(self):
    
        #Market variables
        self.Price = np.zeros((self.C, self.R, self.S, self.T-self.tInit))
        self.Volume = np.zeros((self.C, self.R, self.S, self.T-self.tInit))
        self.returns = np.zeros((self.C, self.R, self.S, self.T-1-self.tInit))
                
        self.dfmacro_list = []
        self.dfHurst_list=[]
        self.dfHurst_adj_list=[]        
        self.dfMSD_list=[]
        self.dfvolatility_list=[]        
        self.dfBubble_list=[]

    def load_macrovariables(self, dataframe_type):
        
        for s in range(self.S):
                    
            sFolder = str(s)
            self.macro_variables(dataframe_type, sFolder, s)

    def returns_dataframe(self, variable, bias, s):
    
        r = np.log(variable[1:]) - np.log(variable[:-1]) #returns
        #r = variable[1:] / variable[:-1] -1 #returns
        biasString = self.dict_num2label[bias]

        dftemp = pd.DataFrame(r, columns=["r"])
        dftemp["condition"] = biasString
        dftemp["seed"] = self.rseed
        dftemp['sim'] = s
        
        self.dfmacro_list.append(dftemp)
        
    def MSD_dataframe(self, variable, bias, s):
    
        biasString = self.dict_num2label[bias]
    
    	# Mean Squared Displacement (MSD)

        msd = self.MSD(variable)

        dftempMSD = pd.DataFrame(data=msd, columns=[r'$MSD$'], index=self.lags)
        dftempMSD.index.name = 'lags'
        dftempMSD = dftempMSD.reset_index()

        dftempMSD["condition"] = biasString
        dftempMSD["seed"] = self.rseed
        dftempMSD['sim'] = s

        self.dfMSD_list.append(dftempMSD)
        
        return msd

    def volatility_dataframe(self, price, bias, s):
        #To be fixed!
        r = np.log(price[1:]) - np.log(price[:-1])
        r = abs(r) #absolute returns for volatility clustering

        PriceWindows = self.sliding_window_view(r, window_shape = 5) #week
        #fun = lambda x: (np.log(x)[1:]-np.log(x)[:-1])**2
        
        vol = np.array([fun(x) for x in PriceWindows])
        
        ac = sm.tsa.acf(vol, nlags=1)[-1]

        dftemp = pd.DataFrame(data=[ac], columns=['$AC_{|r|}$']) #r'$m_{MSD}$'
        dftemp["condition"] = biasString
        dftemp["seed"] = self.rseed
        dftemp['sim'] = s

        self.dfvolatility_list.append(dftemp)    

    def Hurst_dataframe(self, msd, bias, s):

        biasString = self.dict_num2label[bias]

        # Hurst
        
        slope = self.Hurst(msd)
        
        dftemp = pd.DataFrame(data=[slope], columns=[self.hurst_col_name]) #r'$m_{MSD}$'
        dftemp["condition"] = biasString
        dftemp["seed"] = self.rseed
        dftemp['sim'] = s

        self.dfHurst_list.append(dftemp)

    def Hurst_adj_dataframe(self, price, bias, s):

        biasString = self.dict_num2label[bias]

        # Hurst adjusted
        H = self.Hurst_adj(price)
        
        dftemp = pd.DataFrame(data=[H], columns=[self.hurst_col_name]) #r'$m_{MSD}$'
        dftemp["condition"] = biasString
        dftemp["seed"] = self.rseed
        dftemp['sim'] = s

        self.dfHurst_adj_list.append(dftemp)

    def bubble_dataframe(self, Price, bias, s, sFolder):

        fundamentalValueTrue = self.fetch.saved('fundamentalValueTrue', s=sFolder, verbose=False)[0, self.tInit:]
        actionForecasting = self.fetch.saved('actionForecasting', s=sFolder, verbose=False)[:, 0, self.tInit:]

    	# Cumulative distance from fundamental value and price; variance around forecasting action (strategy)
        
        DeltaF = np.mean( (Price / fundamentalValueTrue -1)**2 )
        #DeltaF = np.mean( np.log( Price ) - np.log( fundamentalValueTrue ) )
        #DeltaF = np.log10( np.mean( ( np.log(Price)-np.log(fundamentalValueTrue) )**2 ) )
        
        #DeltaF = np.mean( ( np.log(Price)-np.log(fundamentalValueTrue) )**2 )
        DeltaF = np.mean( abs( (Price-fundamentalValueTrue)/fundamentalValueTrue) )
                
        aFpol = np.mean(actionForecasting, axis=1).var(axis=0) #how strategy varies across agents I
        sigmaaF = np.mean(actionForecasting, axis=0).var(axis=0) #how market varies in time T
        muaF = actionForecasting.mean()
                
        data = [{
                r'$\mu_{\mathcal{F}}$': muaF, #mean strategy overall, in time and across population
                r'$\sigma^2_{\mathcal{F}, T}$': sigmaaF,
                r'$\sigma^2_{\mathcal{F}, I}$': aFpol,
                #r'$\log_{10} {\text{MSI}}$': DeltaF}]
                'MRAI': DeltaF}]
    
        biasString = self.dict_num2label[bias]
    
        dftemp = pd.DataFrame.from_dict(data, orient='columns')
        dftemp["condition"] = biasString
        dftemp["seed"] = self.rseed
        dftemp['sim'] = s

        self.dfBubble_list.append(dftemp)
       
    def macro_variables(self, dataframe_type, sFolder, s):

        Price = self.fetch.saved('Price', s=sFolder, verbose=False)[0, self.tInit:]

        if dataframe_type == 'macro':
            self.returns_dataframe(Price, self.bias, s)
        elif dataframe_type == 'MSD':
            self.MSD_dataframe(Price, self.bias, s)
        elif dataframe_type == 'Hurst':
            msd = self.MSD_dataframe(Price, self.bias, s)
            self.Hurst_dataframe(msd, self.bias, s)  
        elif dataframe_type == 'Hurst_adj':
            self.Hurst_adj_dataframe(Price, self.bias, s) 
        elif dataframe_type == 'volatility':
            self.volatility_dataframe(Price, self.bias, s)          
        elif dataframe_type == 'bubble':
            self.bubble_dataframe(Price, self.bias, s, sFolder)

        # Fundamental Value

        if self.bias == 1 and dataframe_type in ['macro', 'MSD', 'Hurst', 'Hurst_adj']:
       
            fundamentalValueTrue = self.fetch.saved('fundamentalValueTrue', s=sFolder, verbose=False)[0, self.tInit:]
           
            bias = 'fundamental value'
            bias = self.dict_string2num[bias] 
            
            if dataframe_type == 'macro':
                self.returns_dataframe(fundamentalValueTrue, bias=bias, s=s)
            elif dataframe_type == 'MSD':
                self.MSD_dataframe(fundamentalValueTrue, bias=bias, s=s)
            elif dataframe_type == 'Hurst':
                msd = self.MSD_dataframe(fundamentalValueTrue, bias=bias, s=s)
                self.Hurst_dataframe(msd, bias=bias, s=s)
            elif dataframe_type == 'Hurst_adj':
                self.Hurst_adj_dataframe(fundamentalValueTrue, bias=bias, s=s)

    def initialise_microvariables(self):
        
        self.Atotal = np.zeros((self.C, self.R, self.S, self.I, self.T-self.tInit))
        
        self.alphaConfName = r'$\alpha_{conf}$'
        self.alphaDiscName = r'$\alpha_{disc}$'
        self.betaName = r'$\beta$'
        self.tauName = r'$\tau$'
        self.kappaName = r'$\kappa$'
        self.gName = r'$g$'
        self.rhoBidName = r'$\rho_{bid}$'
        
        self.parametersNames = [self.alphaConfName, self.alphaDiscName, self.betaName, self.tauName, self.kappaName, self.gName, self.rhoBidName]
        self.parametersNamesNoBias = [self.alphaConfName, self.betaName, self.tauName, self.kappaName, self.gName, self.rhoBidName]
        
        self.QTName = r'$\pi_{T}$'
        
        self.WName = r'$\Delta \log{W}$'
        #self.WName = r'$\frac{\tilde{W}}{W_0}-1$'
        
        self.dfmicro_list = []
        self.dfQT_list = []
        
    def load_microvariables(self, Q=False):
    
        #RL parameters
        RLparameters = self.fetch.parameters_csv('RL')
        
        #self.beta = RLparameters['beta']
        self.alphaConf = RLparameters['alphaConf']        
        self.alphaDisc = RLparameters['alphaDisc']
        self.beta = RLparameters['beta']
        
        SYMBAparameters = self.fetch.parameters_csv('SYMBA')
        self.tau = SYMBAparameters['tau']
        self.rhoBid = SYMBAparameters['rhoBid']
        self.kappa = SYMBAparameters['kappa']
        self.g = SYMBAparameters['g']
        self.g *= SYMBAparameters['gSkewness']
        
        self.parameters = [self.alphaConf, self.alphaDisc, self.beta, self.tau, self.kappa, self.g, self.rhoBid]
        
        for s in range(self.S):
                    
            sFolder = str(s)
            self.micro_variables(sFolder, s) if not Q else self.Q_variables(sFolder, s)
        
    def micro_variables(self, sFolder, s):      
        
	#self.VolumeOwned[s] = self.fetch.saved('VolumeOwned', s=sFolder, verbose=False)[:, 0, self.tInit:]
	#self.Aequity[s] = self.fetch.saved('Aequity', s=sFolder, verbose=False)[:, 0, self.tInit:]
	#self.Wallet[s] = self.fetch.saved('Wallet', s=sFolder, verbose=False)[:, self.tInit:]
	#self.AvailableMargin[s] = self.fetch.saved('AvailableMargin', s=sFolder, verbose=False)[:, self.tInit:]
        self.Atotal[self.bias, self.rseed, s] = self.fetch.saved('Atotal', s=sFolder)[:, self.tInit:]

        deltalogW = np.log( np.median(self.Atotal[self.bias, self.rseed, s, :, 1:], axis=-1) ) - np.log( self.Atotal[self.bias, self.rseed, s, :, 0] ) #returns
        #deltalogW = ( np.median( self.Atotal[self.bias, self.rseed, s, :, 1:], axis=-1 ) / self.Atotal[self.bias, self.rseed, s, :, 0] ) -1 #returns

        dftemp = pd.DataFrame( deltalogW.reshape(-1), columns=[self.WName])
        dftemp["condition"] = self.bias
        dftemp["seed"] = self.rseed
        dftemp["sim"] = s
        dftemp["ID"] = range(self.I)
        dftemp["ID"] = dftemp["ID"].astype(str)
        dftemp["ID"] += f'-rs{self.rseed}'
        dftemp[self.alphaDiscName] = self.alphaDisc
        dftemp[self.alphaConfName] = self.alphaConf
        dftemp[self.tauName] = self.tau
        dftemp[self.gName] = self.g
        dftemp[self.kappaName] = self.kappa
        dftemp[self.rhoBidName] = self.rhoBid
        dftemp[self.betaName] = self.beta

        self.dfmicro_list.append(dftemp)
        
    def Q_variables(self, sFolder, s):      
                
        dfQT = self.df_QT(sFolder, s)
        self.dfQT_list.append(dfQT)   
        
    def df_QT(self, sFolder, s):
    
        QT = self.fetch.saved('QTrading', s=sFolder)[:, self.tInit:].mean(axis=1)
    
        idx = list(range(self.I))
        idx = [str(i)+f'-rs{self.rseed}' for i in idx]

        #s0 = [r'$T$', r'$F$', r'$M$']
        #s0_order = [0, 1, 2] 
        
        s0 = [r'$T$', r'$F$']
        s0_order = [0, 1] 
        
               
        s0_dict = dict(zip(s0, s0_order))
        
        s1 = [r'$V_{-} \, & \, H \leq P$',
        r'$V_{+} \, & \, P_{avg} \leq H \leq P$',
        r'$V_{+} \, & \, H \leq P_{avg} < P$',
        r'$V_{+} \, & \, H \leq P \leq P_{avg}$',
        
        r'$V_{-} \, & \, P < H$',
        r'$V_{+} \, & \, P_{avg} < P < H$',
        r'$V_{+} \, & \, P \leq P_{avg} \leq H$',
        r'$V_{+} \, & \, P < H < P_{avg} $']
        
        s1_order = [0, 2, 4, 6, 1, 3, 5, 7]        
        s1_dict = dict(zip(s1, s1_order))
        
        action = ['Bid', 'Hold', 'Ask']
        action_order = [0, 1, 2]
        action_dict = dict(zip(action, action_order))
        
        names = ["ID", r"$s_0$", r"$s_1$", r"$a_T$"]
        Qcolumn = r'$Q_T$'
        aColumn = r'$a_T$'
        
        index = pd.MultiIndex.from_product([idx, s0, s1, action], names = names)
        
        x = pd.DataFrame(QT.reshape(-1), index=index, columns=[Qcolumn])

        x.reset_index(inplace=True)
        x[r'$s0_{order}$'] = x[r"$s_0$"].map(s0_dict)
        x[r'$s1_{order}$'] = x[r"$s_1$"].map(s1_dict)
        x[r'$a_{order}$'] = x[r"$a_T$"].map(action_dict)
        x.sort_values(by=[r'$s0_{order}$', r'$s1_{order}$', r'$a_{order}$'], ascending=[True, True, True], inplace=True)
        
        x['beta'] = x.reset_index().ID
        x['beta'] = x.beta.map(dict(zip(x.beta.unique(), self.beta)))
        x[Qcolumn] *= x.beta
        x.drop('beta', axis=1, inplace=True)
                        
        x['exp'] = np.exp(x[Qcolumn])
        x.drop(Qcolumn, axis=1, inplace=True)
        x.reset_index(inplace=True)
        x = x.drop(x[(x[aColumn] == 'Ask')*(x[r'$s_1$']==s1[0])].index)
        x = x.drop(x[(x[aColumn] == 'Ask')*(x[r'$s_1$']==s1[4])].index)
        x = x.drop(x[x[r"$s_0$"]==r'$M$'].index) #drop mixed forecast
        x.set_index(["ID", r'$s0_{order}$', r"$s_0$", r'$s1_{order}$', r"$s_1$", aColumn], inplace=True)

        df = x.unstack(aColumn)['exp']

        df = df.apply(lambda x: x/df.sum(axis=1))
        df = df.stack(aColumn)
        df = df.to_frame().rename(columns={0: self.QTName})
        df.reset_index(inplace=True)

        df[r'$a_{order}$'] = df[aColumn].map(action_dict)
        df.sort_values(by=[r'$s0_{order}$',  r'$s1_{order}$', r'$a_{order}$'], ascending=[True, True, True], inplace=True)

        #for parameterName, parameter in zip(self.parametersNames, self.parameters):
            
        #    df[parameterName] = df.ID
        #    ID2parameter = dict( zip(df[parameterName].unique(), parameter) )
        #    df[parameterName] = df[parameterName].map(ID2parameter)
        
        df["condition"] = self.bias
        df["seed"] = self.rseed
        df["sim"] = s
        
        df.set_index(["condition", "seed", "sim", "ID"], inplace=True)         
        
        return df

def get_parser():
    
    #https://stackoverflow.com/questions/15301147/python-argparse-default-value-or-specified-value
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', nargs='?', const=True, type=int, help = 'please give the comparison number to analyse', required=False, default=-1)
    #parser.add_argument('--datasetEnd', nargs='?', const=True, type=int, help = 'please give the name of a dataset folder', required=False, default='all')
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    
    args = get_parser()
    n = args.n
    
    #create_plot_folders()
    
    CompareAnalysis(n)
