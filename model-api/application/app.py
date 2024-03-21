import statsmodels.iolib.smpickle as smpickle
import pandas as pd
from flask import Flask, request, abort
from feature_engineering import feature_preprocessing
from predictions_and_postprocessing import postprocessed_predictions

app = Flask(__name__)
final_model = smpickle.load_pickle('./final_model.pkl')
selected_features = ['x5','x81', 'x31','x91','x53','x44','x12','x62','x58','x56']
selected_features_string = ','.join(selected_features)
invalid_data_message = f'Invalid data, input data should contain the following columns [{selected_features_string}]'
@app.route('/predict', methods=['POST'])
def return_predictions():
    if request.is_json:
        orig_data = request.json
        orig_data = pd.read_json(orig_data, orient='records')
        if set(set(selected_features)).issubset(set(orig_data.columns)):
            processed_data = feature_preprocessing(orig_data)
            predictions = postprocessed_predictions(final_model, orig_data, processed_data)
            predictions = predictions.to_json(orient='records')
            return predictions
        else:
            abort(415, description=invalid_data_message)
    else:
        abort(415, description='Not Valid JSON')

@app.route('/status', methods=['GET'])
def return_status():
    return("Application is up and serving requests")
