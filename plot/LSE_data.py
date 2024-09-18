import pandas as pd
import os
import glob
from tqdm import tqdm
import numpy as np

# use glob to get all the csv files 
# in the folder
path = os.getcwd()
folderRawData = '/LSE-20120927-20220927/'
csv_files = glob.glob(os.path.join(path+folderRawData, "*.csv"))

folderLSEdata = path+'/LSE_data/'
os.makedirs(folderLSEdata) if not os.path.isdir(folderLSEdata) else None #create LSE data folder for new csv data files, in case it does not exist

#%%

lengths = []

# loop over files to understand optimal length
for f in tqdm( csv_files ):
    
    df = pd.read_csv(f, sep=';')
    lengths.append(len(df))

counts = []
    
[counts.append(lengths.count(item)) for item in set(lengths)]

OptimalLength = list(set(lengths))[np.argmax(np.array(counts))]

#%%

### Extract data of optimal length

SeriesNames = []
SeriesReturns = []
SeriesVolumes = []

# loop over the list of csv files
for f in tqdm( csv_files ):
    
    df = pd.read_csv(f, sep=';')
    
    if len(df) == OptimalLength:
        
        df['Year'] = pd.to_datetime(df.DATE).dt.year
        
        if df.Year.iloc[-1]-df.Year.iloc[0] == 10:
            
            try:
                SeriesReturns.append(df.CLOSE.str.replace(',', '.').astype(float))
            except:
                SeriesReturns.append(df.CLOSE.replace(',', '.').astype(float))
            
            try:
            	 SeriesVolumes.append(df.VOLUME.apply(lambda x: x.split(',')[0]).astype(int) )
            except:
            	 SeriesVolumes.append(df.VOLUME.astype(int))
            #breakpoint()
            #print(df.VOLUME.astype(int))
            
            SeriesNames.append(df.SYMBOL.unique()[0].split('.')[0])

LSEreturns = pd.concat(SeriesReturns, axis=1)
LSEreturns.columns = SeriesNames
LSEreturns.to_csv(folderLSEdata+'LSE_data_closing_prices.csv', index=False)

LSEvolumes = pd.concat(SeriesVolumes, axis=1)
LSEvolumes.columns = SeriesNames
LSEvolumes.to_csv(folderLSEdata+'LSE_data_volumes.csv', index=False)
