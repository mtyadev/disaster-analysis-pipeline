import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import re

def load_data(database_filepath):
    """Loads Data from given database

    Args:
        database_filepath (string): Path to Database to be loaded

    Returns:
        X (series): Series of messages
        Y: (numpy array): Array stating to which category a message belongs
        category_names (list): List of all categories contained in the dataset
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    connection = engine.connect()
    table_name = extract_table_name(database_filepath)
    df = pd.read_sql_table(table_name,connection)
    X = df.message
    Y = np.array(df.loc[:,'related':'direct_report'])
    category_names = df.loc[:,'related':'direct_report'].columns
    return X,Y,category_names

def extract_table_name(database_filename):
    """Extracts a name of a table from a given database_filename

    Args:
        database_filename (string): Database from which table name is extracted
    """
    table_name = re.search("([^/]+)(\.db)",database_filename)
    return table_name[1]


def tokenize(text):
    """Splits text into single words (tokens). Afterwards cleans tokens.
    Reduces all tokens to their dictionary form by lemmatizing them.
    Removes whitespaces and casts all tokens to lowercase.

    Args:
        text (string): Text to be tokenized and cleaned

    Returns:
        list: list of cleaned tokens
    """
    tokens = word_tokenize(text)

    # Cleaning tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """Builds a machine learning model pipeline. Using RandomForestClassifier
    and applying grid search to find the best estimator by tuning different
    hyperparameters.

    Args:
        None

    Returns:
        model: Best model based on executed cross validation
    """
    randomforest = RandomForestClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(randomforest))
    ])

    # To find best hyperparameter tuning parameters checked:
    # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    parameters = {'clf__estimator__bootstrap': [True, False],
                 #'clf__estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                 #'clf__estimator__max_features': ['auto', 'sqrt'],
                 'clf__estimator__min_samples_leaf': [1, 2, 4],
                 'clf__estimator__min_samples_split': [50, 100, 150],
                 'clf__estimator__n_estimators': [5, 10]
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates performance of given model for all given categories

    Args:
        model: Machine Learning model to be evaluated
        X_test (np_array): X values for given test set
        Y_test (np_array): Y values for given test set
    """

    Y_pred = model.predict(X_test)

    for counter,element in enumerate(category_names):
        print(category_names[counter].format())
        print(classification_report(Y_test[counter],Y_pred[counter]))

def save_model(model, model_filepath):
    """Pickles given model at given filepath

    Args:
        model: Machine Learning model to be pickled
        model_filepath: Destination where model should be pickled
    """

    # Checked how to pickle model from a pipeline
    # https://stackoverflow.com/questions/34143829/sklearn-how-to-save-a-model-created-from-a-pipeline-and-gridsearchcv-using-jobli
    from sklearn.externals import joblib
    joblib.dump(model.best_estimator_, model_filepath, compress = 1)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
