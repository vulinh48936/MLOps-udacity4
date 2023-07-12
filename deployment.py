import os
import sys
import shutil
import logging
import json

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def store_model_into_pickle():
    """
    Duplicate the most recent model pickle file, the value from "latestscore.txt",
    and the "ingestfiles.txt" file and move them to the deployment directory.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = config['output_folder_path']
    prod_deployment_path = config['prod_deployment_path']
    model_path = config['output_model_path']

    logging.info("Deploying trained model to production")

    logging.info("Copying trainedmodel.pkl, ingestfiles.txt, and latestscore.txt")
    shutil.copy(os.path.join(model_path, 'trainedmodel.pkl'), prod_deployment_path)
    shutil.copy(os.path.join(dataset_csv_path, 'ingestedfiles.txt'), prod_deployment_path)
    shutil.copy(os.path.join(model_path, 'latestscore.txt'), prod_deployment_path)

if __name__ == '__main__':
    store_model_into_pickle()
