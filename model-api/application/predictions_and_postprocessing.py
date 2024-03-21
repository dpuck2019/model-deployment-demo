import pandas as pd

def predictions(final_model, processed_data):
    """Takes in the data, drops the Y column and uses the passed model to make predictions. Then returns a dataframe with
    predictions labeled phat

    Args:
        final_model (statsmodels.api.logit): The previously trained model
        processed_data (pandas DataFrame): A dataframe containing the preprocessed data
    Returns:
        pandas DataFrame: A dataframe containing the predicted probabilities, labeled as phat
    """    
    predictions = pd.DataFrame(final_model.predict(processed_data)).rename(columns={0:'phat'})
    return predictions

def postprocessing(orig_data, model_results):
    """Takes the predicions, adds a predicted class based on the cutoff provided by business partners, then sorts the original
    inputs and combines the inputs and predictions

    Args:
        orig_data (pandas DataFrame): A dataframe containing the original inputs provided to the API
        model_results (pandas DataFrame): A dataframe containing the predicted probabilities from the model

    Returns:
        pandas DataFrame: A dataframe containing the sorted original columns, the predicted probabilities, and the predicted class
    """    
    model_results.loc[model_results['phat'] >= 0.75, 'business_outcome'] = 'Event'
    model_results.loc[model_results['phat'] < 0.75, 'business_outcome'] = 'Non-Event'
    orig_data.sort_index(axis=1, inplace=True)
    model_results = pd.concat([orig_data, model_results], axis=1, sort=False)
    return model_results

def postprocessed_predictions(final_model, orig_data, processed_data):
    """A function to take in the processed data, and the model and return the original values, predicted probabilities, and business outcome

    Args:
        final_model (statsmodel.api.logit): The trained model
        orig_data (pandas DataFrame): The original unprocessed data passed to the API
        processed_data (pandas DataFrame): The data that has been processed by the feature_engineering step

    Returns:
        pandas DataFrame: A dataframe containing the sorted original columns, predicted probabilities, and business outcome
    """    
    pred = predictions(final_model, processed_data)
    postprocessed_predictions = postprocessing(orig_data, pred)
    return postprocessed_predictions  

