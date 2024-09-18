import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from inequalipy import gini
import string

from filemanagement.store import FileStorage
from filemanagement.fetch import FileFetching

from plot.palette_colorblind import palette, conditions_order

import seaborn as sns
import matplotlib.pyplot as plt

import os

import matplotlib as mpl
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial']})
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

# Save files so we do not need to create them every time, but only upload them!

n = 12

store = FileStorage()
fetch = FileFetching(store)
comparison, n = fetch.comparison(n)

ComparisonFolder = fetch.plotFolderPath+f'/comparison_{n}'

exStart = comparison['exStart']
    
CognitiveTraits = comparison['cognitiveTraits']        
randomSeeds = comparison['randomSeeds']        
C = len(CognitiveTraits)
R = len(randomSeeds)

N = R*C
        
datasets = range(exStart, exStart+N)

store = FileStorage()
store.set_example(exStart)
fetch = FileFetching(store)

plotPath = fetch.plotFolderPath+f'/comparison_{n}/Figures/'

# Mapping between cognitive traits and abbreviations
cognitive_traits = ['optimistic bias', 'no bias', 'pessimistic bias', 'no bias individual', 'no learning']
conditions = [r'$B_{+}$', r'$B_{0}$', r'$B_{-}$',  r'$B_{00}$', r'$Z$']
biasDict = dict(zip(cognitive_traits, conditions))

palette = palette[:5]
sns.set_palette(palette)

# Define the column name for wealth log gap
wealth_name = r"$\log{W_{\text{total}, T_{\text{testing}}}} - \log{W_{\text{total}, T_{training}}}$"

# Define the reference condition you want to compare against (r"$B_{+}$")
reference_condition = r"$B_{+}$"

tInit = 253

def create_df(N, exN, phase='testing'):
    df_list = []
        
    # Loop over random seeds
    for n in range(N):
        store = FileStorage()
        store.set_example(exN + n)
        fetch = FileFetching(store)
        
        # Fetch input parameters once
        input_params = fetch.input_parameters()
        random_seed = input_params['randomSeed']
        condition = input_params['CognitiveTraits'][0]
        condition_abbr = biasDict[condition]  # Abbreviation for the current condition
        I = input_params['I']  # Number of agents
        
        if phase == 'testing':
            # Fetch Winit once (log10 of wealth at the end of training)
            Winit = np.log10(fetch.saved('Atotal', s='training_0')[:, -1])
        else:
            Price_tInit = fetch.saved('Price', s='training_0')[0, tInit-1]
            VolumeOwned_tInit = fetch.parameters_json('wallet')['InitialVolumeOwned']
            Liquidity_balance = fetch.parameters_json('wallet')['Liquidity balance']
            Winit = Price_tInit*VolumeOwned_tInit*(1+Liquidity_balance)
            Winit = np.log10(Winit)
        
        # Get number of simulations once
        S = fetch.input_parameters(time=True)['S']
        
        # Loop over simulations
        for s in range(S):
            # Fetch W (log10 of wealth at the end of testing) and compute the log wealth gap
            W = np.log10(fetch.saved('Atotal', s=str(s))[:, -1]) - Winit
            
            # Create a temporary DataFrame for this simulation
            df_temp = pd.DataFrame({
                wealth_name: W,
                'condition': condition_abbr,
                'random_seed': random_seed,
                'sim': s,
                'agent': np.arange(I)  # Indexing agents
            })
            
            # Append the DataFrame for this simulation to the list
            df_list.append(df_temp)
    
    # Concatenate all the DataFrames
    df = pd.concat(df_list)

    return df   
    
def create_dfW(N, exN, phase='testing'):
    df_list = []
        
    # Loop over random seeds
    for n in range(N):
        store = FileStorage()
        store.set_example(exN + n)
        fetch = FileFetching(store)
        
        # Fetch input parameters once
        input_params = fetch.input_parameters()
        random_seed = input_params['randomSeed']
        condition = input_params['CognitiveTraits'][0]
        condition_abbr = biasDict[condition]  # Abbreviation for the current condition
        I = input_params['I']  # Number of agents
        
        # Get number of simulations once
        S = fetch.input_parameters(time=True)['S']
        
        # Loop over simulations
        for s in range(S):
        
            if phase == 'testing':
                # Fetch W (log10 of wealth at the end of testing)
                W = fetch.saved('Atotal', s=str(s))[:, -1]
                #W = np.log10(fetch.saved('Atotal', s=str(s))[:, -1])
            else:
                W = fetch.saved('Atotal', s='training_0')[:, -1]
                #W = np.log10(fetch.saved('Atotal', s='training_0')[:, -1])
                            
            # Create a temporary DataFrame for this simulation
            df_temp = pd.DataFrame({
                'wealth': W,
                'condition': condition_abbr,
                'random_seed': random_seed,
                'sim': s,
                'agent': np.arange(I)  # Indexing agents
            })
            
            # Append the DataFrame for this simulation to the list
            df_list.append(df_temp)
    
    # Concatenate all the DataFrames
    df = pd.concat(df_list)

    return df     
    
def create_dfsummary(df):    
    
    # Group by condition, random seed, and simulation to compute summary statistics
    dfgrouped = df.groupby(['condition', 'random_seed', 'sim'])
    
    # Create a summary DataFrame with the desired statistics
    dfsummary = pd.DataFrame()
    dfsummary[r'$\mu_W$'] = dfgrouped[wealth_name].mean()
    dfsummary[r'$\log{ \sigma_W^2 }$'] = np.log(dfgrouped[wealth_name].var())
    dfsummary[r'$Skew_W$'] = dfgrouped[wealth_name].apply(lambda x: skew(x, bias=False))
    dfsummary[r'$Kurt_W$'] = dfgrouped[wealth_name].apply(lambda x: kurtosis(x, bias=False))
    
    return dfsummary

def create_df_pivot(df, wealth_name):
    """
    Run the Friedman test on the full dataset without averaging over agents and simulations.
    """
    # We do NOT group by random_seed and average anymore.
    # Instead, we keep each agent-simulation pair as a separate repeated measure.
    
    # Pivot the data so each row corresponds to an agent-simulation pair
    df_pivot = df.pivot_table(
        index=['random_seed', 'sim', 'agent'],  # Retain the full structure
        columns='condition',                    # Conditions as columns
        values=wealth_name                   # The log wealth gap values
    )
    
    # Drop rows with any missing data (if some agents are not tested under certain conditions)
    df_pivot = df_pivot.dropna()
    
    return df_pivot

def run_friedman_test_on_full_data(df_pivot, wealth_log_gap):
    """
    Run the Friedman test on the full dataset without averaging over agents and simulations.
    """
    # We do NOT group by random_seed and average anymore.
    # Instead, we keep each agent-simulation pair as a separate repeated measure.
    
    # The test expects a separate array for each condition
    conditions_data = [df_pivot[condition] for condition in df_pivot.columns]
    
    friedman_stat, friedman_pvalue = friedmanchisquare(*conditions_data)
    
    return friedman_stat, friedman_pvalue

def posthoc_wilcoxon(df_pivot, alpha=0.05):
    """
    Perform post-hoc Wilcoxon signed-rank tests for all pairwise condition comparisons
    after a significant Friedman test result.
    
    df_pivot: DataFrame with conditions as columns and agent-simulation pairs as rows
    alpha: significance level for Bonferroni correction
    """
    # List of condition pairs to compare
    conditions = df_pivot.columns
    condition_pairs = list(combinations(conditions, 2))  # Get all pairs of conditions
    
    # Initialize a dictionary to store results
    posthoc_results = {}
    
    # Number of comparisons for Bonferroni correction
    num_comparisons = len(condition_pairs)
    
    # Bonferroni-corrected significance level
    corrected_alpha = alpha / num_comparisons
    
    # Loop over all condition pairs and run Wilcoxon signed-rank test
    for cond1, cond2 in condition_pairs:
        # Extract the data for the two conditions
        data_cond1 = df_pivot[cond1]
        data_cond2 = df_pivot[cond2]
        
        # Perform Wilcoxon signed-rank test
        stat, pvalue = wilcoxon(data_cond1, data_cond2)
        
        # Store the results with Bonferroni-corrected p-value
        posthoc_results[(cond1, cond2)] = {
            'statistic': stat,
            'pvalue': pvalue,
            'significant': pvalue < corrected_alpha  # Mark if the result is significant
        }
    
    return posthoc_results, corrected_alpha

def posthoc_wilcoxon_against_Bplus(df_pivot, alpha=0.05, target_condition=r"$B_{+}$"):
    """
    Perform post-hoc Wilcoxon signed-rank tests for condition comparisons against r"$B_{+}$"
    after a significant Friedman test result.
    
    df_pivot: DataFrame with conditions as columns and agent-simulation pairs as rows
    alpha: significance level for Bonferroni correction
    target_condition: the condition to which all others will be compared (default is B_{+})
    """
    # List of conditions to compare against the target condition
    conditions = df_pivot.columns
    other_conditions = [cond for cond in conditions if cond != target_condition]
    
    # Initialize a dictionary to store results
    posthoc_results = {}
    
    # Number of comparisons for Bonferroni correction
    num_comparisons = len(other_conditions)
    
    # Bonferroni-corrected significance level
    corrected_alpha = alpha / num_comparisons
    
    # Loop over other conditions and run Wilcoxon signed-rank test against the target condition
    for cond in other_conditions:
        # Extract the data for the target condition and the current condition
        data_target = df_pivot[target_condition]
        data_cond = df_pivot[cond]
        
        # Perform Wilcoxon signed-rank test
        stat, pvalue = wilcoxon(data_target, data_cond)
        
        # Store the results with Bonferroni-corrected p-value
        posthoc_results[(target_condition, cond)] = {
            'statistic': stat,
            'pvalue': pvalue,
            'significant': pvalue < corrected_alpha  # Mark if the result is significant
        }
    
    return posthoc_results, corrected_alpha


##########################################

#df = create_df(N, exN)

#df = create_dfW(N, exN)

#df_pivot = create_df_pivot(df, wealth_log_gap)

fig, axs = plt.subplots(1, 2, figsize=(6,6)) #, sharex=True, sharey=False)

ax1 = axs[0]

filepath = ComparisonFolder+'/df_pivot_wealth.csv'
        
index = ['random_seed', 'sim', 'agent']
y = r"Wealth ($\log_{10}$)"

if os.path.isfile(filepath):

    df_pivot = pd.read_csv(filepath)
    df_pivot = df_pivot.set_index(index)
else:

    df = create_dfW(N, exStart)
    column_name = 'wealth' #df.columns[0]
    df = df.rename(columns={column_name: y})
    df_pivot = df.pivot_table(index=index, columns='condition', values=y)
    df_pivot = np.log10(df_pivot)

    friedman_stat, friedman_pvalue = run_friedman_test_on_full_data(df_pivot, y)
    print(f"Friedman Test: Statistic = {friedman_stat:.5f}, p-value = {friedman_pvalue:.5f}")

    res = posthoc_wilcoxon_against_Bplus(df_pivot, alpha=0.05, target_condition=r"$B_{+}$")
    df_stats = pd.DataFrame.from_dict(res[0], orient='index')
    print(df_stats)

    df_pivot = df_pivot[conditions_order[:5]]
    contrasting_conditions = df_pivot.columns[1:]
    df_diff = df_pivot.subtract(df_pivot.iloc[:, 0], axis=0)[contrasting_conditions].median()

    df_diff.to_csv(ComparisonFolder+'/df_median_diff_wealth.csv')
    
    df_pivot.to_csv(filepath)

### Subplot 1

sns.violinplot(df_pivot, alpha=.25, inner=None, ax=ax1)
#sns.stripplot(df_pivot, alpha=.5, ax=ax1)
sns.boxplot(df_pivot, fill=False, whis=0, showfliers=False, ax=ax1, color=[.5, .5, .5])
sns.stripplot(df_pivot.median().to_frame().T, color='k', ax=ax1, label='Median', s=8.5)
ax1.set_ylabel(y)
ax1.set_ylim([6.25, 8.05])
#ax1.set_ylim([0, 8.05])

y, h, col = 7.75, 0.035, 'k'

x1, x2 = 0, 1
ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax1.text((x1+x2)*.5, y+h/2, "*", ha='center', va='bottom', color=col, size=12)

y, h, col = 7.75+0.035, 0.085, 'k'

x1, x2 = 0, 2
ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax1.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=12)

y, h, col = 7.75+0.035+0.085, 0.085, 'k'

x1, x2 = 0, 3
ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax1.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=12)

y, h, col = 7.75+0.035+2*0.085, 0.085, 'k'

x1, x2 = 0, 4
ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax1.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=12)

ax1.text(-0.15, 1.065, string.ascii_lowercase[0], transform=ax1.transAxes, size=12, weight='bold')

############## Inequality ###################
 
df_pivot_variable_name = "Gini index"

filepath = ComparisonFolder+'/df_pivot_inequality.csv'

index = ['random_seed', 'sim']
y = "Gini index"

if os.path.isfile(filepath):

    df_pivot = pd.read_csv(filepath)
    df_pivot = df_pivot.set_index(index)
    
else:

    df = create_dfW(N, exStart)
    column = 'wealth' #df.columns[0]

    dfgini = df.groupby(['condition','random_seed', 'sim'])[column].apply(lambda x: gini(x))
    dfgini = dfgini.to_frame()
    dfgini = dfgini.rename(columns={column: y})
    df_pivot = dfgini.pivot_table(index=index, columns='condition', values=df_pivot_variable_name)

    df_pivot = df_pivot[conditions_order[:5]]
    df_pivot.to_csv(filepath)

    friedman_stat, friedman_pvalue = run_friedman_test_on_full_data(df_pivot, y)
    print(f"Friedman Test: Statistic = {friedman_stat:.5f}, p-value = {friedman_pvalue:.5f}")

    res = posthoc_wilcoxon_against_Bplus(df_pivot, alpha=0.05, target_condition=r"$B_{+}$")
    df_stats = pd.DataFrame.from_dict(res[0], orient='index')
    print(df_stats)

    contrasting_conditions = df_pivot.columns[1:]
    df_diff = df_pivot.subtract(df_pivot.iloc[:, 0], axis=0)[contrasting_conditions].median()
    df_diff.to_csv(ComparisonFolder+'/df_median_diff_inequality.csv')
 
### Subplot 2

ax = axs[1]

sns.violinplot(df_pivot, alpha=.25, inner=None, ax=ax)
sns.stripplot(df_pivot, alpha=.35, ax=ax)
sns.boxplot(df_pivot, fill=False, showfliers=False, whis=0, ax=ax, color=[.5, .5, .5])
sns.stripplot(df_pivot.median().to_frame().T, color='k', ax=ax, label='Median', s=8.5)
ax.set_ylabel(y)

y, h, col = df_pivot.max().max(), 0.045, 'k'

x1, x2 = 0, 3
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=12)

y, h, col = df_pivot.max().max(), 0.10, 'k'

x1, x2 = 0, 4
ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
ax.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=12)

ax.text(-0.15, 1.065, string.ascii_lowercase[1], transform=ax.transAxes, size=12, weight='bold')

plt.tight_layout()

plt.savefig(plotPath+'WI.pdf', bbox_inches='tight')
