from flask import Flask, jsonify, request
import pandas as pd
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list
from scoring import score_model

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

config = load_config()
dataset_csv_path = os.path.join(config['output_folder_path'])
prediction_model = None

@app.route('/')
def index():
    return "Hello World"

@app.route("/prediction", methods=['GET', 'POST'])
def predict():        
    """
    Create a prediction endpoint that loads data from a file path and invokes
    the prediction function defined in "diagnostics.py".
    Returns:
        json: Predictions made by the model in JSON format.
    """

    filepath = request.args.get('filepath')

    df = pd.read_csv(filepath)
    df = df.drop(['corporation', 'exited'], axis=1)

    preds = model_predictions(df)
    preds_to_list = preds.tolist()
    return jsonify(preds_to_list)

@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    """
    Create a scoring endpoint
    that executes the "scoring.py" script and obtains the score of the deployed model.
    Returns:
        str: model f1 score
    """

    f1 = score_model()
    f1_str = str(f1)
    return f1_str

@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    """
    Create a summary statistics endpoint that invokes the
    DataFrame summary function defined in "diagnostics.py".
    Returns:
        json: summary statistics
    """
    stat = dataframe_summary()
    return jsonify(stat)

@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    """
    Create a diagnostics endpoint that invokes the following
    functions defined in "diagnostics.py":
    Returns:
        dict: missing_percentage
              execution_time
              outdated_package_list
    """

    exc_time = execution_time()
    missing_percentage = missing_data()
    outdated_packages = outdated_packages_list()
    result = {
        'execution_time': exc_time,
        'missing_percentage': missing_percentage,
        'outdated_packages': outdated_packages
    }
    
    return jsonify(result)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
