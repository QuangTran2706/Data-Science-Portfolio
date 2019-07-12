# -*- coding: utf-8 -*-
# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import pickle
import joblib
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

def load_data(database_filepath):
    """
        Load data from the sqlite database. 
    Args: 
        database_filepath: the path of the database file
    Returns: 
        X (DataFrame): messages 
        Y (DataFrame): One-hot encoded categories
        category_names (List)
    """    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_message', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X, Y , category_names

def tokenize(text):
     # remove non-alphebetic contents, including punctuations, numbers
    clean_tokens = []
    for sent in text:
        # remove non-alphebetic contents, including punctuations, numbers
        sent = re.sub(r"[^a-zA-Z]", " ", sent)
        tokens = word_tokenize(sent)
        lemmatizer = WordNetLemmatizer()
        # lemmatize and remove stop words
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok.lower()).strip()
            if clean_tok not in stopwords.words("english"):
                clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
      build NLP pipeline - tf-idf vectorizer, multiple output classifier,
      grid search the best parameters
    Args: 
        None
    Returns: 
        cross validated classifier object
    """     
    pipeline = Pipeline([
    ('vect', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(XGBClassifier(n_estimators = 100, colsample_bytree=0.5)))
    ])
    parameters = [{'vect__max_features': [80000] , #[50000, 80000],
              'clf__estimator__max_depth': [15] #[10,15]
              }]

    grid_cv = GridSearchCV(pipeline, parameters, cv=3, verbose=2, n_jobs = 1)
    return grid_cv

def model_evaluation(model, x_test, y_test, avg_param, category_names ):
    """
    output model performance as a dataframe
    """
    y_pred = model.predict(x_test)
    # build classification report on every column
    performances = []
    for i in range(len(category_names)):
        performances.append([f1_score(y_test[:, i], y_pred[:, i], average= avg_param),
                             precision_score(y_test[:, i], y_pred[:, i], average= avg_param),
                             recall_score(y_test[:, i], y_pred[:, i], average= avg_param)])
        # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],
                                    index = category_names)   
    return performances

def export_model(model, model_filepath):
    # Export model as a pickle file
    #with open(model_filepath, 'wb') as f:
        #pickle.dump(model, f)
    joblib.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model_pipeline = build_model()
        
        print('Training model...')
        model_pipeline.fit(x_train, y_train)
        best_model = model_pipeline.best_estimator_
        
        print('Evaluating model...')
        model_evaluation(best_model, x_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        export_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the 2 arguments in order:'\
              '1. filepath of the disaster messages database '\
              '2. filepath of the pickle file to save the model '\
              '\n\nExample: python train_classifier.py DisasterResponse.db disaster_response_classifier.pkl')


if __name__ == '__main__':
    main()
