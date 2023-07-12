import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def merge_multiple_dataframe():
    """
    Create a data ingestion function that checks for the presence of datasets, merges them, removes duplicate entries,
    and saves information about the ingested files to "ingestedfiles.txt"
    along with the ingested data to "finaldata.csv".
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']

    df = pd.DataFrame()
    file_names = []

    logging.info(f"Reading files from {input_folder_path}")
    for file in glob.glob(os.path.join(input_folder_path, '*.csv')):
        temp_df = pd.read_csv(file)

        # Save filenames
        file_names.append(file)
        # Merge datasets
        df = df.append(temp_df, ignore_index=True)

    logging.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    logging.info("Saving ingested metadata")
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write(f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(file_names))

    logging.info("Saving ingested data")
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)


if __name__ == '__main__':
    merge_multiple_dataframe()