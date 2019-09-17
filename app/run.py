import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/pickled_modle.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Generate Total Messages by Category Data
    category_counts = df.iloc[:,4:].apply(pd.Series.value_counts)
    category_counts = category_counts.iloc[1,:].tolist()
    category_names = df.iloc[:,4:].columns.tolist()

    # Generate Total Messages by Genre by Category Data
    direct_messages = []
    news_messages = []
    social_messages = []

    def messages_by_genre(genre,category_list):
        """Analyzes the sum of messages belonging to a given genre for each
        category in the dataset.

        Args:
            genre (string): messages are categorized as 'direct','news','social'
            category_list (list): contains all available categories in dataset
        """

        sum_messages_by_genre = []
        for category in category_list:
            sum_messages_by_category = df[(df['genre'] == genre) & (df[category] == 1)][category].sum()
            sum_messages_by_genre.append(sum_messages_by_category)
        return sum_messages_by_genre

    direct_messages = messages_by_genre('direct',category_names)
    news_messages = messages_by_genre('news',category_names)
    social_messages = messages_by_genre('social',category_names)

    # Own Graph
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Messages by Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                     Bar(name='direct', x=category_names, y=direct_messages),
                     Bar(name='news', x=category_names, y=news_messages),
                     Bar(name='social', x=category_names, y=social_messages),
            ],

            'layout': {
                'title': 'Messages by Categories and Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }


    ]


    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
