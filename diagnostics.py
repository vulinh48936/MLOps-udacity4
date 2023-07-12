import os
import sys
import json
import pickle
import timeit
import logging
import subprocess
import numpy as np
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def model_predictions(X):
    """
    Parameters:
        X (pandas.DataFrame): A DataFrame containing the input features.
    Returns:
        predictions: Predictions made by the model.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    prod_deployment_path = config['prod_deployment_path']

    logging.info("Loading model")
    model = pickle.load(open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb'))

    logging.info("Running predictions")
    predictions = model.predict(X)
    return predictions

def dataframe_summary():
    """
    Calculate summary statistics: mean, median, and standard deviation on numerical data.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_path = config['output_folder_path']

    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_path, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df = data_df.select_dtypes('number')

    logging.info("Calculating statistics for data")
    statistics_dict = {}
    for col in data_df.columns:
        mean = data_df[col].mean()
        median = data_df[col].median()
        stdv = data_df[col].std()
        statistics_dict[col] = {'mean': mean, 'median': median, 'stdv': stdv}

    return statistics_dict

def missing_data():
    """
    Calculates percentage of missing data for each column.
    Returns:
        list[dict]: Percentage of missing values for each column.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_path = config['output_folder_path']

    logging.info("Loading and preparing finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_path, 'finaldata.csv'))

    logging.info("Calculating missing data percentage")
    missing_data = data_df.isna().sum()
    percentage = missing_data / data_df.shape[0] * 100
    missing_list = {col: {'percentage': perc} for col, perc in zip(data_df.columns, percentage)}
    return missing_list

def execution_time():
    """
    Calculate timing of ingestion.py and training.py.
    """
    logging.info("Calculating time for ingestion.py and training.py")
    ingestion_time = []
    training_time = []
    for _ in range(20):
        starttime = timeit.default_timer()
        os.system('python ingestion.py')
        time = timeit.default_timer() - starttime
        ingestion_time.append(time)

        starttime = timeit.default_timer()
        os.system('python training.py')
        time = timeit.default_timer() - starttime
        training_time.append(time)

    mean_of_ingest_time = sum(ingestion_time) / len(ingestion_time)
    mean_of_train_time = sum(training_time) / len(training_time)

    return [mean_of_ingest_time, mean_of_train_time]

def outdated_packages_list():
    """
    Examine the status of dependencies listed in the "requirements.txt" file using "pip-outdated",
    which evaluates each package to determine if it requires an update.
    Returns:
        str: Standard output generated by the "pip-outdated" command.
    """
    logging.info("Checking outdated dependencies")
    dependencies = subprocess.run(
        ['pip-outdated', 'requirements.txt'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8'
    )

    dep = dependencies.stdout
    return dep

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    test_data_path = config['test_data_path']
    X = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X = X.drop(['corporation', 'exited'], axis=1)
    model_predictions(X)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
