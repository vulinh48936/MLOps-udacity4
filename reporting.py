import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def score_model():
    """
    Compute the confusion matrix for the test data using the deployed model,
    and generate a visualization of the matrix.
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    dataset_csv_path = config['output_folder_path']
    model_path = config['output_model_path']

    logging.info("Loading and preparing testdata.csv")
    test_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y_true = test_df.pop('exited')

    logging.info("Predicting test data")
    X = test_df.drop(['corporation'], axis=1)
    prediction = model_predictions(X)
    cf_matrix = confusion_matrix(y_true, prediction)

    logging.info("Plotting and saving confusion matrix")
    sns.heatmap(cf_matrix, annot=True)
    plt.savefig("confusionmatrix.png")

if __name__ == '__main__':
    score_model()
