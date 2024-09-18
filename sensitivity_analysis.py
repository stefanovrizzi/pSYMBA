import numpy as np
import pandas as pd

from emukit.core import ContinuousParameter, DiscreteParameter, ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.core.initial_designs.latin_design import LatinDesign
from emukit.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity, MonteCarloSensitivity 

import GPy
from GPy.models import GPRegression
from emukit.model_wrappers import GPyModelWrapper

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

import main
import multiprocessing as mp

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching

import pickle
import json
import string

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib import ticker
import seaborn as sns

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# Define system

def run_main(n):

    x = X[n]
    x[0] *= 250
    x[0] += 21
    
    hyperparameters = dict( zip( parameter_names, x ) )
    print(hyperparameters, '\n')
    
    exampleNumber = exStart+n
    
    main.Main(hyperparameters_to_tweak=hyperparameters, exampleNumber=exampleNumber)  

def parallelise_simulations(index):

    pool = mp.Pool(processes=12)
    pool.map(run_main, index)
    pool.close()
    pool.join()

### Sensitivity analysis

def GP_sensitivity(x_, y_, readout):
    
    print(readout)

    # Normalize the data
    #scaler = StandardScaler()
    #x_ = scaler.fit_transform(x_)
    
    #scaler = RobustScaler()
    
    #Only for diagnostics
    #y_[:, 0] = x_[:, 2]
    
    ### Fit gaussian process
    #kernel = GPy.kern.RBF(ARD=True, input_dim=np.shape(x_)[1])
    kernel = GPy.kern.RBF(ARD=True, input_dim=np.shape(x_)[1], lengthscale=100) #lengthscale=100
    #kernel = GPy.kern.Exponential(input_dim=np.shape(x_)[1])
    #kernel = GPy.kern.Matern32(input_dim=np.shape(x_)[1])

    #try:    
    #    model_gpy = GPRegression(x_, y_, kernel)
    #except:
    #    breakpoint()
        
    model_gpy = GPRegression(x_, y_, kernel)
        
    #try:
    #    model_emukit = GPyModelWrapper(model_gpy)
    #except:
    #    breakpoint()

    model_emukit = GPyModelWrapper(model_gpy)

    model_emukit.optimize()

    ### Compute sensitivity

    sensitivity = MonteCarloSensitivity(model=model_emukit, input_domain=space)

    try:
        main_effects, total_effects, _ = sensitivity.compute_effects(num_monte_carlo_points=num_monte_carlo_points)
    except:
        breakpoint()
        kernel = GPy.kern.RBF(ARD=False, input_dim=np.shape(x_)[1], lengthscale=10)
        
    # Create pandas DataFrame for Gaussian Process method
    df_main = create_df(main_effects, readout)
    df_total = create_df(total_effects, readout)
    
    return df_main, df_total

def create_df(data, readout):

    data = {ivar: data[ivar][0] for ivar in data}
    df = pd.DataFrame({'value': data, 'readout': readout})

    # Reset index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'parameter'}, inplace=True)

    return df

def input_parameters():

    store = FileStorage()
    fetch = FileFetching(store)

    store.set_example(exStart)
    fetch = FileFetching(store) 

    #Input parameters        
    InputTimeParameters = fetch.input_parameters(time=True)
    T = InputTimeParameters['T']
    S = InputTimeParameters['S']
        
    SYMBAParameters = fetch.parameters_json('SYMBA')
    tInit = SYMBAParameters['tInit']
    
    return S, tInit

def prepare_readouts(index):

    df_list = []

    df_MSD_list = []

    for n in range(N):
        
        store = FileStorage()
        store.set_example(exStart+n)
        fetch = FileFetching(store)  
                    
        for s in range(S):
                    
            sFolder = str(s)
        
            Price = fetch.saved('Price', s=sFolder, verbose=False)[0, tInit:]
            
            df = pd.DataFrame(data=Price, columns=['Price'])
            df['n'] = n
            df['s'] = s
            df['t'] = np.ones(len(df)).cumsum().astype(int)
            df_list.append(df)
            
            msd_slope = MSD(Price)
            df_MSD_list.append(msd_slope)

    dfcon = pd.concat(df_list)
    dfcon.set_index(['t', 'n', 's'], inplace=True)
    dfcon = dfcon.unstack(['n', 's'])['Price']
    dfcon = np.log(dfcon).diff(periods=1, axis=0).iloc[1:] #log returns 
    
    lags = np.logspace(4, 30, base=1.2).astype(int) #int(self.T/2))
    lags = list( set(lags) )
    sliding_window_view = np.lib.stride_tricks.sliding_window_view
    
    df = pd.DataFrame()
    df[r'$\mu$'] = dfcon.mean()
    df[r'$\sigma^2$'] = dfcon.var()
    df[r'$Skew$'] = dfcon.skew()
    df[r'$Kurt$'] = pd.DataFrame.kurt(dfcon)
        
    df[r'$AC_{d}$'] = dfcon.apply(lambda x : sm.tsa.acf(x, nlags=1)[-1])
    df[r'$AC_{m}$'] = dfcon.apply(lambda x : sm.tsa.acf(x, nlags=21)[-1])
    df[r'$AC_{Y}$'] = dfcon.apply(lambda x : sm.tsa.acf(x, nlags=252)[-1])
    
    df[r'$m_{MSD}$'] = df_MSD_list
    
    return df

def Sobol(df):

    # Ys to evaluate
    readouts = df.columns
    R = len(readouts)
    Y = np.zeros((N*S, R))
    
    x = np.repeat(X, repeats=S, axis=0)
    y = np.zeros((N*S, 1))
    
    df_main_list = []
    df_total_list = []
    
    for r, readout in enumerate(readouts):
        Y[:, r] = df[readout].to_numpy()

        y[:, 0] = Y[:, r]
        #y[:, 0] = np.random.normal(100, 0.1, size=N*S)
        #x[:, 0] = y[:, 0]*1
        #x[:, 1] = y[:, 0]*2
        #x[:, 2] = np.random.normal(10, 1, size=N*S)
        
        main_effects, total_effects = GP_sensitivity(x, y, readout)
        main_effects['readout'] = readout
        total_effects['readout'] = readout
        df_main_list.append(main_effects)
        df_total_list.append(total_effects)
    
    #Y = df.to_numpy()
    #main_effects, total_effects = GP_sensitivity(X, Y, readouts[0])
        
    df_main = pd.concat(df_main_list)
    df_total = pd.concat(df_total_list)
    
    return df_main, df_total
    
def plot_Sobol(df, effect):

    color = 'k'

    readouts = df.readout.unique()

    fig, axs = plt.subplots(nrows=int(len(readouts)/2), ncols=2, figsize=(8, 5), sharex=True, sharey=True)
    axs = axs.flatten()
    
    for r, readout in enumerate(readouts):
        
        ax = axs[2*r -7*(r>3)]
        
        df_temp = df[df.readout == readout]
        sns.barplot(data=df_temp, x='parameter', y='value', hue='effect', palette=['r', 'b'], ax=ax)
        #sns.barplot(data=df_temp, x='parameter', y='value', color=color, ax=ax[r])
        ax.set_ylabel('')

        ax.legend(title='effect') if r == 0 else ax.get_legend().remove()

        # Add subplot label
        ax.text(-0.12, 1.065, string.ascii_lowercase[r], transform=ax.transAxes, size=12, weight='bold')

    ylabel = 'Explained output variance (fraction of total output variance)'
    fig.text(0.065, 0.5, ylabel, va='center', rotation='vertical')
    
    plt.savefig(f'output/plots/Sobol_{effect}_effect.png', dpi=300)
    plt.close()       

def plot_singleparameter(data, dfy, readout):

    data = np.repeat(data, repeats=S, axis=0)

    df = pd.DataFrame(data=data, columns=parameter_names)
    #df = df.melt()
    df[readout] = dfy[readout].values

    #g = sns.FacetGrid(df, x_vars=df.columns[:-1], y_vars='color', height=3, aspect=1.5)
    #g = g.map(plt.scatter, alpha=.5)

    fig, axs = plt.subplots(figsize=(12, 6), nrows=2, ncols=int( len(df.columns[:-1])/2 ), sharey=True)
    axs = axs.flatten()
    
    for ax, x in zip(axs, df.columns[:-1]):
    
        sns.scatterplot(data=df, x=x, y=readout, ax=ax)
        x_ = df[x].values
        y_ = df[readout].values
        unique_x = np.unique(x_)    
        ax.plot(unique_x, np.poly1d(np.polyfit(x_, y_, 1))(unique_x), color='k', linewidth=2)
        
        #mu = np.median(y_)
        #ax.set_ylim([ max(y_.min(), mu-2*y_.std()), min(y_.max(), mu+2*y_.std()) ])

    plt.tight_layout()

    plt.savefig(f'output/plots/singleparameter_{readout}.pdf')
    plt.close()  

def plot_lincoeff(x_, dfy):
    
    x_ = np.repeat(x_, repeats=S, axis=0)
    
    # Normalize the data
    scaler = StandardScaler()
    x_ = scaler.fit_transform(x_)

    fig, axs = plt.subplots(figsize=(20, 8), nrows=2, ncols=4, sharex=True)
    axs = axs.flatten()
    
    for ax, readout in zip(axs, dfy.columns):
    
        y_ = dfy[readout].to_numpy()
        y_ = y_.reshape(-1, 1)
        #y_ = scaler.fit_transform(y_)

        #scaler = RobustScaler()
        #y_ = scaler.fit_transform(y_)
    
        regressor = LinearRegression()
        regressor.fit(x_, y_) 
        
        dfcoeff = pd.DataFrame(data=regressor.coef_, columns=parameter_names)
        dfcoeff = dfcoeff.rename(columns=dfdict)
        dfcoeff = dfcoeff.melt()
        dfcoeff = dfcoeff.rename(columns={'variable': 'parameter'})

        ax.axhline(0, color=[.5, .5, .5], alpha=.5)
        darkred = [.5, 0, 0]
        darkblue = [0, 0, .5]
        grey = [.5, .5, .5]
        g = sns.barplot(data=dfcoeff, x='parameter', y='value', palette=['k', 'k', 'k', 'k', darkred, darkred, darkblue, darkblue, darkblue, darkblue, grey, grey, grey, grey], ax=ax) #color='k'
        ax.set_ylabel(readout)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        # Get the minimum and maximum values across all subplots
        data_min = dfcoeff.value.min()
        data_max = dfcoeff.value.max()
        
        if (data_min < 0)*(data_max>0):

            data_min *= 1.05
            data_max *= 1.05

            data_max = max(abs(data_min), data_max)  
            data_min = -max(abs(data_min), data_max)
            
        elif (data_min > 0)*(data_max>0):
        
            data_min *= 0.95
            data_max *= 1.05

        elif (data_min < 0)*(data_max<0):
        
            data_min *= 1.05
            data_max = 0

        ax.set_ylim([data_min, data_max])
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    plt.savefig(f'output/plots/lincoeff.png', dpi=300)
    plt.close()  

def plot_pairwise(data, dfy, readout):

    df = pd.DataFrame(data=data, columns=parameter_names)
    df['color'] = dfy[readout].values
    
    cmap = 'coolwarm'
        
    #g = sns.PairGrid(df, vars=df.columns, hue='color', palette='coolwarm_r', diag_sharey=False, corner=True)
    #g = g.map_lower(plt.scatter)

    g = sns.PairGrid(df, x_vars=df.columns[:-2], y_vars=df.columns[1:-1], hue='color', palette=cmap, height=3, aspect=1.5)
    g = g.map(plt.scatter, alpha=.5)

    # Get the minimum and maximum values across all subplots
    data_min = df['color'].values.min()
    data_max = df['color'].values.max()
    
    if (data_min < 0)*(data_max>0):

        data_max = max(abs(data_min), data_max)    
        data_min = -max(abs(data_min), data_max)

    # Get the current axes
    ax = plt.gca()

    # Create a scatter plot outside of the grid to generate colorbar
    scatter = ax.scatter([], [], c=[], cmap=cmap, vmin=data_min, vmax=data_max)

    # Get the colorbar
    cbar = plt.colorbar(scatter)

    # Set label for the colorbar
    cbar.set_label(readout)

    plt.savefig(f'output/plots/pairwise_{readout}.png', dpi=300)
    plt.close()       

def MSD(variable):

    # Mean Squared Displacement (MSD)

    SquaredPriceResiduals = np.zeros( len(lags) )

    for tn, t_ in enumerate(lags):
        
        PriceWindows = sliding_window_view(variable, window_shape = t_)
        
        price_residuals = lambda x: (np.log(x[-1]) - np.log(x[0]))**2
        
        try:
            SquaredPriceResiduals[tn] = np.array([price_residuals(x) for x in PriceWindows]).mean()
        except:
            breakpoint()

    result = linregress(np.log( lags ), np.log( SquaredPriceResiduals ))
    slope = result.slope
        
    return slope

def get_ex_index():

    store = FileStorage(verbose=False)
    store.output_folders()

    store.set_flag()
    exStart = store.flag #+1

    store.set_flag( exStart+N )

    del store

    index = list( range(N) )
       
    return exStart, index

################## Main ##########################

load = True #False
path = 'output/data/X'
spacePath = 'output/data/space.pkl'

space = ParameterSpace([
                        #DiscreteParameter('taumax', list(range(10, 252))),
                        ContinuousParameter('taumax', 0, 1), #rescaled live!
                        ContinuousParameter('rhoBidMax', 0.1, 1),
                        ContinuousParameter('kappamax', 0.0, 1),
                        ContinuousParameter('gShape', 1, 6),
                        #]) #,
                        ##ContinuousParameter('aAlphaConf', 1, 10),
                        ##ContinuousParameter('bAlphaConf', 1, 10),
                        ##ContinuousParameter('aAlphaDisc', 1, 10),
                        ##ContinuousParameter('bAlphaDisc', 1, 10),
                        ContinuousParameter('alphaConfMax', 0, 1),
                        ContinuousParameter('alphaDiscMax', 0, 1),
                        ContinuousParameter('nu_plus_d', 0.01, 2.0),
                        ContinuousParameter('nu_minus_d', 0.01, 0.99),
                        ContinuousParameter('nu_zero_d', 0.01, 0.10),
                        ContinuousParameter('etaHalf', 0.01, 0.1),
                        ContinuousParameter('sigma', 0.005, 0.05),
                        ContinuousParameter('RiskFreeRate', 0.0, 0.05),
                        ContinuousParameter('RiskPremium', 0.0, 0.05),
                        ContinuousParameter('b', 0.0, 0.01)
                        ])

N = 500 #number of points
num_monte_carlo_points = 10000

#design = RandomDesign(space)
design = LatinDesign(space)
parameter_names = design.parameter_space.parameter_names

lags = np.logspace(4, 30, base=1.2).astype(int) #int(self.T/2))
lags = list( set(lags) )
sliding_window_view = np.lib.stride_tricks.sliding_window_view

if load:

    # Load the space object from the file
    with open(spacePath, 'rb') as file:
        space = pickle.load(file)

    X = np.load(path+'.npy')
    exStart = 7629
    index = range(N)


    #for n in range(N):
        
    #    store = FileStorage()
    #    store.set_example(exStart+n)
    #    fetch = FileFetching(store)  
                    
    #    for p, parameter in enumerate(parameter_names):

    #        X[n, p] = fetch.parameters_json('SYMBA')[parameter]

else:
        
    # Save the space object to a file
    with open(spacePath, 'wb') as file:
        pickle.dump(space, file)

    X = design.get_samples(N)
    np.save(path, X)

    exStart, index = get_ex_index()
    
    data = {'exStart': exStart, 'N': N}
    with open(f'output/data/sensitivity_{exStart}_N_{N}', 'w') as f: # An arbitrary collection of objects supported by pickle.    
        json.dump(data, f) # Pickle the 'data' dictionary using the highest protocol available

    
    parallelise_simulations(index)

S, tInit = input_parameters()
df = prepare_readouts(index)

#fig, axs = plt.subplots(ncols=len(df.columns), figsize=(16, 6))
#[df.hist(column=column, ax=ax) for column, ax in zip(df.columns, axs)], plt.grid(False), plt.tight_layout(), plt.show()

#scaler = QuantileTransformer(output_distribution='normal', n_quantiles=100)
#y_ = scaler.fit_transform(y_)

scaler = StandardScaler()
df[r'$\mu$'] = scaler.fit_transform(df[r'$\mu$'].to_numpy().reshape(-1, 1))

scaler = PowerTransformer()
df[r'$\sigma^2$'] = scaler.fit_transform(df[r'$\sigma^2$'].to_numpy().reshape(-1, 1))

scaler = PowerTransformer()
df[r'$Skew$'] = scaler.fit_transform(df[r'$Skew$'].to_numpy().reshape(-1, 1))

scaler = PowerTransformer()
df[r'$Kurt$'] = scaler.fit_transform(df[r'$Kurt$'].to_numpy().reshape(-1, 1))

#scaler = StandardScaler()        
df[r'$AC_{d}$'] = scaler.fit_transform(df[r'$AC_{d}$'].to_numpy().reshape(-1, 1))
df[r'$AC_{m}$'] = scaler.fit_transform(df[r'$AC_{m}$'].to_numpy().reshape(-1, 1))
df[r'$AC_{Y}$'] = scaler.fit_transform(df[r'$AC_{Y}$'].to_numpy().reshape(-1, 1))
    
df[r'$m_{MSD}$'] = scaler.fit_transform(df[r'$m_{MSD}$'].to_numpy().reshape(-1, 1))

#fig, axs = plt.subplots(ncols=len(df.columns), figsize=(16, 6))
#[df.hist(column=column, ax=ax) for column, ax in zip(df.columns, axs)], plt.grid(False), plt.tight_layout(), plt.show()

dfdict = {'taumax': r'$\tau_{MAX}$',
          'rhoBidMax': r'$\rho_{MAX}$',
          'kappamax': r'$\kappa_{MAX}$',
          'gShape': r'$g$',
          'alphaConfMax': r'$\alpha^{+}_{MAX}$',
          'alphaDiscMax': r'$\alpha^{-}_{MAX}$',
          'nu_plus_d': r'$\nu^{+}_d$',
          'nu_minus_d': r'$\nu^{-}_d$', 
          'nu_zero_d': r'$\nu^{0}_d$',           
          'etaHalf': r'$\eta_{50}$',
          'RiskFreeRate': r'$r_{free}$',
          'RiskPremium': r'$r_{prem}$',
          'sigma': r'$\sigma_{\epsilon}$',
          'b': r'$b$'
          }

plot_lincoeff(X, df)
#[plot_singleparameter(X, df, readout) for readout in df.columns]
#[plot_pairwise(X, df, readout) for readout in df.columns]

df_main, df_total = Sobol(df)

df_main['effect'] = r'$1^{\text{st}}$'
df_total['effect'] = 'Total'

df_effects = pd.concat([df_main, df_total])

df_effects['parameter'] = df_effects['parameter'].map(dfdict)

effect = 'joint'
#plot_Sobol(df_effects, effect)


#effect = 'main'
#plot_Sobol(df_main, effect)

#effect = 'total'
#plot_Sobol(df_total, effect)
