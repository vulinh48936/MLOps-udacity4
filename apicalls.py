import requests
import os
import json
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

config = load_config()
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])

URL = "http://127.0.0.1:8000/"

# Call each API endpoint and store the responses
logging.info(f"Sending POST request to /prediction for {os.path.join(test_data_path, 'testdata.csv')}")
response_prediction = requests.get(URL + f"/prediction?filepath={os.path.join(test_data_path, 'testdata.csv')}").text

logging.info("Sending GET request to /scoring")
response_score = requests.get(URL + '/scoring').text

logging.info("Sending GET request to /summarystats")
response_stat = requests.get(URL + '/summarystats').text

logging.info("Sending GET request to /diagnostics")
response_diagnostics = requests.get(URL + '/diagnostics').text

# Write the responses to a text file
logging.info("Generating report text file")
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as file:
    file.write('Model Predictions\n')
    file.write(response_prediction)
    file.write('\nModel Score\n')
    file.write(response_score)
    file.write('\nStatistics Summary\n')
    file.write(response_stat)
    file.write('\nDiagnostics Summary\n')
    file.write(response_diagnostics)
