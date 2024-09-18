import os
import csv
import pandas as pd

def collect_warnings_log(root_dir='output/data'):
    data = []

    for ex_folder in os.listdir(root_dir):
        if ex_folder.startswith('ex'):
            ex_number = int(ex_folder[2:])
            ex_path = os.path.join(root_dir, ex_folder)

            for sim_folder in os.listdir(ex_path):
                sim_path = os.path.join(ex_path, sim_folder)
                log_file_path = os.path.join(sim_path, 'warnings_log.txt')

                if os.path.isfile(log_file_path):
                    with open(log_file_path, 'r') as log_file:
                        for line in log_file:
                            line = line.strip()
                            if line:  # Ensure it's not an empty line
                                data.append({'ex': ex_number, 'sim': sim_folder, 'warning': line})

    return data

def save_to_csv(data, output_file='output/data/warnings_summary.csv'):
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# Collect warnings log data
data = collect_warnings_log()

# Save the collected data to a CSV file
save_to_csv(data)

