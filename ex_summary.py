import os
import json
import pandas as pd
import numpy as np

# Load flag.npy to get the upper limit of the folder numbers
flag = np.load("output/data/flag.npy")

ex_start = 0

files = ["master", "master_time", "market_parameters", "wallet_parameters", "SYMBA_parameters", "RL_parameters"]

# Initialize a list to store dictionaries from master.json files
data_list = []

for i in range(ex_start, flag.item() + 1):
    folder_name = f"ex{i}"
    folder_path = os.path.join("output/data", folder_name)

    # Check if the folder exists
    if os.path.exists(folder_path):

        data_example = dict()
        data_example.update({'ex': i})
    
        for file_ in files:
    
            # Load master.json
            json_file_path = os.path.join(folder_path, file_+'.json')

            # Check if master.json exists
            if os.path.exists(json_file_path):
        
                with open(json_file_path, "r") as json_file:
                    data = json.load(json_file)
         
                data_example.update(data)
                    
        data_list.append(data_example)
          
                #data_list.append(data)           

                #keys.extend( data.keys() ) if i == ex_start else None                    

df = pd.DataFrame(data=data_list)

# Save DataFrame to CSV
df.to_csv("output/data/ex_summary.csv", sep=';', index=False)
#df.to_excel("output/data/ex_summary.xlsx", index=False)

