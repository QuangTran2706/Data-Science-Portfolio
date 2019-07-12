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


app = Flask(__name__, template_folder='templates')

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('disaster_message', engine)

# load model
model = joblib.load("disaster_response_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # plot 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #plot 2
    category_names = list(df.iloc[:,4:].columns)
    direct_category_count = df.loc[df.genre=='direct',category_names].sum().sort_values(ascending=False)/df.loc[df.genre=='direct',category_names].shape[0]
    direct_category_name = list(direct_category_count.index)
    
    #plot 3
    news_category_count = df.loc[df.genre=='news',category_names].sum().sort_values(ascending=False)/df.loc[df.genre=='news',category_names].shape[0]
    news_category_name = list(news_category_count.index)
    
    #plot 4   
    social_category_count = df.loc[df.genre=='social',category_names].sum().sort_values(ascending=False)/df.loc[df.genre=='social',category_names].shape[0]
    social_category_name = list(social_category_count.index)
    # create visuals
   
    graphs = [
            # plot 1: overall distribution of diff genres of messages
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
              
                 # plot 2: % of direct messages in each category
              {
            'data': [
                Bar(
                    x=direct_category_name,
                    y=direct_category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Categories for Direct Messages',
                'yaxis': {
                    'title': "% Total Direct Messages"
                },
                'xaxis': {
                    #'title': "Message Categories",
                    'tickangle':60
                }
            }
        },
              # plot 3: % of news messages in each category
              {
            'data': [
                Bar(
                    x= news_category_name,
                    y= news_category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Categories for News Messages',
                'yaxis': {
                    'title': "% Total News Messages"
                },
                'xaxis': {
                    #'title': "Message Categories",
                    'tickangle':60
                }
            }
        }, 
              # plot 4: % of social messages in each category
              {
            'data': [
                Bar(
                    x= social_category_name,
                    y= social_category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Categories for Social Messages',
                'yaxis': {
                    'title': "% Total Social Messages"
                },
                'xaxis': {
                    #'title': "Message Categories",
                    'tickangle':60
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
