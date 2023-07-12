import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import logging
import sys
import glob
from ingestion import merge_multiple_dataframe
import re
import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def main():
    config = load_config()

    prod_deployment_path = os.path.join(config['prod_deployment_path']) 
    model_path = os.path.join(config['output_model_path'])
    input_folder_path = os.path.join(config['input_folder_path'])
    data_path = os.path.join(config["output_folder_path"])

    logging.info("Checking data")

    # First, read ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n') for line in file.readlines()[1:]}

    # Second, determine whether the source data folder has files that aren't
    # listed in ingestedfiles.txt
    source_files = []
    for name in glob.glob(input_folder_path + '/*.csv'):
        source_files.append(os.path.basename(name))

    # Deciding whether to proceed, part 1
    # If you found new data, you should proceed. Otherwise, end the process here
    new_data = set(source_files) != set(ingested_files)

    # Ingesting new data
    logging.info("Ingesting new data")
    if new_data:
        merge_multiple_dataframe()

    # Checking for model drift
    logging.info("Checking for model drift")

    # Check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:
        deployed_score = re.findall(r'\d*\.?\d+', file.read())[0]
        deployed_score = float(deployed_score)

    dataset = pd.read_csv(os.path.join(data_path, 'finaldata.csv'))
    y_pred = dataset.pop('exited')
    X = dataset.drop(['corporation'], axis=1)

    prediction = diagnostics.model_predictions(X)
    new_score = f1_score(prediction, y_pred)

    # Deciding whether to proceed, part 2
    logging.info(f"Deployed score = {deployed_score}")
    logging.info(f"New score = {new_score}")

    # If you found model drift, you should proceed. Otherwise, end the process here
    if new_score >= deployed_score:
        logging.info("No model drift occurred")
        return None

    # Re-training
    logging.info("Re-training model")
    training.train_model()
    logging.info("Re-scoring model")
    scoring.score_model()

    # Re-deployment
    logging.info("Re-deploying model")

    # If you found evidence for model drift, re-run the deployment.py script
    deployment.store_model_into_pickle()

    # Diagnostics and reporting
    logging.info("Running diagnostics and reporting")

    # Run diagnostics.py and reporting.py for the re-deployed model
    reporting.score_model()

    # Diagnostics
    os.system("python apicalls.py")


if __name__ == '__main__':
    main()
