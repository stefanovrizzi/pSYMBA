################ Import key functions ############################

from main import Main

from filemanagement.store import FileStorage

import pandas as pd
import multiprocessing as mp

################ Parallel computation parameters #################

def run_job(idx):

    randomSeed, cognitivetrait, exampleNumber = idx

    cognitivetrait = [cognitivetrait]

    #main.Main(hyperparameters_to_tweak, cognitivetrait, randomSeed, exampleNumber)
    Main(CognitiveTraits=cognitivetrait, randomSeed=randomSeed, exampleNumber=exampleNumber)

####################################################################################

if __name__ == '__main__':

    randomSeeds = list(range(700, 1700, 50)) #list(range(0, 500, 50)) #[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, ] #[0, 50, 100, 150, 200]

    cognitiveTraits = ['optimistic bias', 'pessimistic bias', 'no bias', 'no bias individual', 'no learning']

    store = FileStorage(verbose=False)
    store.output_folders()

    store.set_flag()
    exStart = store.flag #+1

    store.comparison(randomSeeds, cognitiveTraits, exStart)        

    index = pd.MultiIndex.from_product([randomSeeds, cognitiveTraits], names=['randomSeed', 'cognitiveTrait']).tolist()
    index = [idx + (exStart+n,) for n, idx in enumerate(index)]

    store.set_flag( exStart+len(index) )

    del store

    mp.set_start_method('spawn')

    pool = mp.Pool(processes=15)
    pool.map(run_job, index)
    pool.close()
    pool.join()

#num_jobs = len(index)  # Number of jobs to run
#processes = []

#num_cores = 14  # Number of available cores    
    
# Create a pool of workers
#with mp.Pool(processes=num_cores) as pool:
    # Create a list of arguments for each job
#    job_args = [job_id for job_id in index]
        
    # Use pool.starmap to run jobs in parallel with multiple arguments
#    pool.starmap(run_job, job_args)
