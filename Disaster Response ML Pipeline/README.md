# Disaster Response Data Pipeline Project

### Project Overview:
In this project, I built an ETL - Machine Learning pipeline that transforms the text data sent from real disaster events and then classifies the categories of the messages.
The results from the project can be used to expedite the disaster relief actions. 
The pipeline is eventually built into a flask application where an emergency worker can input a new message and get classification results in several categories. The landing page of the webapp also includes 4 visualizations built with plotly.

### File Description:
- ETL Pipeline Preparation.ipython: merge, clean, transform the 2 csv data sources and output a database file of a clean dataframe.
- Machine Learning Preparation.ipython: a pipeline is built to vectorize text data with TFIDF and run a classifier on transformed text data to fine tune the parameters, finally evaluates its performance.
- model/ETL_Pipeline.py: ipython notebook turned python file. run the ETL jobs, prep database file for modeling.
- model/train_classifier.py: ipython notebook turned python file. fit, predict, evaluate and export the model to a pickle file.
- flask_app.py: a web app that can classify the messages and show the categoies they belong to with visuals built with plotly.

### Instructions to run .py locally:
Run the following commands in the project's root directory.
 - To run ETL_Pipeline.py, use the following arguments:
 `python ETL_Pipeline.py data/messages.csv data/categories.csv DisasterResponse.db`
 - To run train_classifier.py, use the following arguments:
 `python train_classifier.py DisasterResponse.db disaster_response_classifier.pkl`
 - To run the flask_app.py: `python flask_app.py`. Then go to `localhost:3001/index`.
 
 ![App Snippet] (https://github.com/lalago31/Data-Science-Portfolio/blob/master/Disaster%20Response%20ML%20Pipeline/flask%20app/app%20snippet.PNG)
 
