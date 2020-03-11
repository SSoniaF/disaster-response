### Disaster Response Pipeline Project

## 1. Installation 

Libraries used in the project:
- Machine Learning Libraries: NumPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Web App and Data Visualization Libraries: Flask, Plotly
Version of python: Python 3.6.3

## 2. Project Motivation 
This project is a project assignment of the Udacity course "Data Science", module "Data Engineering".
It aims at practicing tools and techniques to build ETL pipelines, machine learning pipelines for natural language processing and integration with web apps.

## 3.File Descriptions 
The relevant file are in the folder "data". The data to train the models consist of a set of about 26k text messages plus the assigned categories (labeled data for supervised learning).  

## 4. How To Interact With the Project 

To interact with the models (get a prediction of the categories for a text you can enter):

    -Run the following commands in the project's root directory to set up your database and model.
        To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/adaboost.pkl

    -Run the following command in the app's directory to run your web app. python run.py

    -Go to http://0.0.0.0:3001/

For a deep dive about the process to design the ETL and ML pipelines please check the corresponding Jupyter Notebooks

## 5.Licensing, Authors, Acknowledgements 

- Thanks to Figure Eight for providing the labled messages dataset
- Other sources I used to improve the models are mentioned in the Jupyter Notebooks