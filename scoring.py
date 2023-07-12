import os
import sys
import pickle
import pandas as pd
from sklearn import metrics
import json
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def score_model():
    """
    Load a pre-trained model and test data, compute the F1 score of the model on the test data,
    and save the score to the "latestscore.txt" file.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = config['output_folder_path']
    test_data_path = config['test_data_path']
    model_path = config['output_model_path']

    logging.info("Loading testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    logging.info("Loading the trained model")
    model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))

    logging.info("Preparing the test data")
    y_true = test_df.pop('exited')
    X_cor = test_df.drop(['corporation'], axis=1)

    logging.info("Predicting the test data")
    prediction = model.predict(X_cor)
    f1 = metrics.f1_score(y_true, prediction)
    print(f"f1 score = {f1}")

    logging.info("Saving scores")
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1}")
    
    return f1

if __name__ == '__main__':
    score_model()
