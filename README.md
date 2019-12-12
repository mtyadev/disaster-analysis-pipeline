# Disaster Analysis Pipeline

### Project Motivation

Building a Machine Learning based ETL pipeline analyzing incoming text messages which are sent in the event of natural disasters.

The aim is to provide a machine-learning based filter mechanism, consolidating the vast amount of incoming data in the disaster event. The algorithm should understand which incoming messages are related to which type of required aid so that help organizations can coordinate their efforts more efficiently.

This project was done as part of the <a href="https://www.udacity.com/course/data-scientist-nanodegree--nd025">udacity data scientist nano degree programm</a> show-casing the following approaches:

* Building an ETL Pipeline
* Building a Machine Learning Pipeline

### Software Requirements

* Python 3.x
* Numpy & Pandas
* Jupyter
* Scikit Learn

### Installations:

1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores it in the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements

* This project is part of udacity's <a href="https://www.udacity.com/course/data-scientist-nanodegree--nd025">data scientist nano degree programm</a>


