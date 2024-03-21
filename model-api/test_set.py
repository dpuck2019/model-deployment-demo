import unittest
import pandas as pd
import json
import requests

class TestPredictAPI(unittest.TestCase):
    API_URL =  "http://localhost:1313/predict"
    
    # Helper method to send a POST request to the API
    def send_request(self, data):
        headers = {'Content-type':'application/json'}
        return requests.post(self.API_URL, headers=headers, data=data)

    def test_response_status_code(self):
        input_data = pd.read_csv('../exercise_26_test.csv')
        for size in [1, 10, 100, 1000, input_data.shape[0]]:
            test_data = input_data.iloc[:size]
            test_data = json.dumps(test_data.to_json(orient='records'))
            response = self.send_request(test_data)
            self.assertEqual(response.status_code, 200)

    def test_response_columns(self):
        input_data = pd.read_csv('../exercise_26_test.csv')
        for size in [1, 10, 100, 1000, input_data.shape[0]]:
            test_data = input_data.iloc[:size]
            test_data = json.dumps(test_data.to_json(orient='records'))
            response = self.send_request(test_data)
            response_data = pd.DataFrame.from_dict(response.json())
            self.assertTrue("business_outcome" in response_data.columns)
            self.assertTrue("phat" in response_data.columns)

    def test_response_X_columns_order(self):
        input_data = pd.read_csv('../exercise_26_test.csv')
        test_data = input_data.copy(deep=True)
        test_data = json.dumps(test_data.to_json(orient='records'))
        response = self.send_request(test_data)
        response_data = pd.DataFrame.from_dict(response.json())
        X_cols = response_data.drop(['business_outcome', 'phat'], axis=1)
        self.assertTrue(set(input_data.columns) == set(X_cols.columns))
        self.assertTrue(all(X_cols.columns[i] <= X_cols.columns[i + 1] for i in range(len(X_cols.columns) - 1)))

    def test_missing_column_error(self):
        input_data = pd.read_csv('../exercise_26_test.csv')
        missing_data = input_data.drop('x62', axis=1)
        missing_data = json.dumps(missing_data.to_json(orient='records'))
        missing_response = self.send_request(missing_data)
        self.assertEqual(missing_response.status_code, 415)

    def test_non_JSON_request(self):
        non_JSON_response = requests.post(self.API_URL, data="input_data")
        self.assertEqual(non_JSON_response.status_code, 415)

if __name__ == '__main__':
    unittest.main()
