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
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    connection = engine.connect()
    table_name = extract_table_name(database_filepath)
    df = pd.read_sql_table(table_name,connection)
    X = df.message
    Y = np.array(df.loc[:,'related':'direct_report'])
    category_names = df.loc[:,'related':'direct_report'].columns
    return X,Y,category_names

def extract_table_name(database_filename):
    table_name = re.search("([^/]+)(\.db)",database_filename)
    return table_name[1]


def tokenize(text):
    tokens = word_tokenize(text)

    # Cleaning tokens
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    randomforest = RandomForestClassifier()
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(randomforest))
    ])

    parameters = {'clf__estimator__bootstrap': [True, False],
                 'clf__estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                 #'clf__estimator__max_features': ['auto', 'sqrt'],
                 'clf__estimator__min_samples_leaf': [1, 2, 4],
                 'clf__estimator__min_samples_split': [50, 100, 150],
                 'clf__estimator__n_estimators': [5, 10]
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for counter,element in enumerate(category_names):
        print(category_names[counter].format())
        print(classification_report(Y_test[counter],Y_pred[counter]))

def save_model(model, model_filepath):
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
