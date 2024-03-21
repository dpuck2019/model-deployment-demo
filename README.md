# model-deployment-demo
This project takes a pre-trained generalized linear model and then creates a REST API to return predictions when sent JSON data at port 1313. This API can accept any range of rows at once in the format of the test_set csv located in this repo. As long as docker and python are installed, one only needs to run `python run_api.py` to get this model up and running.
