from sklearn.neighbors import KernelDensity
from scipy import stats

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LSEstockSelection:

    def __init__(self):
    
        self.LSE_data()
        self.LSE_selection()
        #self.negative_log_likelihood()
        #self.plot_NLL()
        
        kernel = stats.gaussian_kde(self.LSEselection.T) #covariance=self.LSEselection.T.cov()
        self.NormalisedNegativeLogLikelihood = kernel.logpdf(self.LSEselection.T)
        self.NormalisedNegativeLogLikelihood = pd.DataFrame(data=self.NormalisedNegativeLogLikelihood.T, columns=['log(pdf)'], index=self.LSEr.columns)
        self.NormalisedNegativeLogLikelihood.index.name = 'Stock'
        self.NormalisedNegativeLogLikelihood.reset_index(inplace=True)
        
        top = self.NormalisedNegativeLogLikelihood.sort_values(by='log(pdf)', ascending=False).iloc[:10]
        
        plt.figure(figsize=(10, 6))
        
        ax = plt.subplot(121)
        sns.barplot(data=top, x='log(pdf)', y='Stock', ax=ax)
        ax.set_title(f"Bandwidth={kernel.covariance_factor()*self.LSEselection.std()}")
        
        LSE = self.LSE[top['Stock']].unstack().reset_index()
        LSE.rename({'level_0':'Stock', 'level_1':'Trading day', 0:'Price'}, axis=1, inplace=True)
        
        ax = plt.subplot(122)
        sns.lineplot(data=LSE, x='Trading day', y='Price', hue='Stock', ax=ax)
        #ax.set_title(f"Bandwidth={kernel.set_bandwidth('scott')}")     
        
        plt.tight_layout()
        plt.show()
        
    def LSE_selection(self):

        LSEselection = pd.DataFrame()
        LSEselection['mean'] = self.LSEr.mean().T
        LSEselection['var'] = self.LSEr.var().T
        LSEselection['skew'] = self.LSEr.skew().T
        LSEselection['kurt'] = self.LSEr.kurt().T
        sliding_window = np.lib.stride_tricks.sliding_window_view    
        LSEselection['mean_5d_volatility'] = self.LSEr.apply( lambda x : sliding_window(x, window_shape = 5).var(axis=1).mean() )
        LSEselection['AC1'] = self.LSEr.apply(lambda x : sm.tsa.acf(x, nlags=1)[1])

        LSEselection = LSEselection.apply(lambda x:(x-min(x))/(max(x)-min(x)), axis = 0) #scale between 0 and 1
        self.LSEselection = LSEselection - LSEselection.mean() #center on 0

    def LSE_data(self):

        ### Empirical LSE data ###
        try:
            self.LSE = pd.read_csv('LSE_data/LSE_data_closing_prices.csv', index_col=False)
        except:
            self.LSE = pd.read_csv('plot/LSE_data/LSE_data_closing_prices.csv', index_col=False)
        #breakpoint()
        
        stockSelection = (self.LSE.diff(periods=1, axis=0).iloc[1:] == 0).mean() < .5 #active at least more than 50% of times
        stockSelection = pd.DataFrame(stockSelection, columns=['Active'])
        stockSelection = stockSelection[stockSelection['Active']==True].index
        
        self.LSEr = np.log10(self.LSE[stockSelection]).diff(periods=1, axis=0).iloc[1:]

    def stock_selection_criteria(self):
    
        LSEselection[ (LSEselection['mean']>0) * (LSEselection['mean']<4*10**(-4)) *
         (LSEselection['var']<10**(-4)) *
         (LSEselection['kurt']<10**(1.5)) * (LSEselection['skew']<0) ].index.tolist()
         
        return LSEselection
         
    def negative_log_likelihood(self):
    
        bandwidths = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100]
        self.NormalisedNegativeLogLikelihood = np.zeros(( len(self.LSEselection), len(bandwidths) ))
        #best = []
            
        for b, bandwidth in enumerate(bandwidths):
            
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.LSEselection)
            self.NormalisedNegativeLogLikelihood[:, b] = - kde.score_samples(self.LSEselection)
                        
            #breakpoint()
            
            #self.NormalisedNegativeLogLikelihood[b] = self.NormalisedNegativeLogLikelihood[b] / self.NormalisedNegativeLogLikelihood[b].max()
            #best.extend( np.argwhere(self.NormalisedNegativeLogLikelihood[b] == 1) )
            
            #best = np.argwhere(kde.score_samples(self.LSEselection) == kde.score_samples(self.LSEselection).min())[0]
        
        breakpoint()
        #self.NormalisedNegativeLogLikelihood = self.NormalisedNegativeLogLikelihood / np.min(self.NormalisedNegativeLogLikelihood, axis=0) - 1
        self.NormalisedNegativeLogLikelihood = pd.DataFrame(data=self.NormalisedNegativeLogLikelihood.T, columns=self.LSEr.columns, index=bandwidths)
        self.NormalisedNegativeLogLikelihood.index.name = 'kernel bandwidth'
        self.NormalisedNegativeLogLikelihood.columns.name = 'stock abbreviation'
        
    def plot_NLL(self):
        
        score = self.NormalisedNegativeLogLikelihood.sum()
        ranking = score.sort_values(ascending=True).index
        self.NormalisedNegativeLogLikelihood = self.NormalisedNegativeLogLikelihood[ranking]
        
        plt.figure(figsize=(18, 3))
        sns.heatmap(self.NormalisedNegativeLogLikelihood)

        plt.title(' '.join(ranking.tolist()[:5]))
        plt.tight_layout()
        plt.show()
        
        
if __name__ == '__main__':
    
    LSEstockSelection()

