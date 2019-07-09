# -*- coding: utf-8 -*-
# import packages
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Arg: file path of data source messages.csv and categories.csv
    Return: clean merged dataframe of messages and categories
    '''
    # 1. read in file
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
        
    # 2. clean data   
    categories_2 = categories.categories.str.split(';', expand=True)
    row = categories_2.iloc[1,:]
    category_colnames = row.apply(lambda x: x[:-2])
        # rename the columns of `categories`
    categories_2.columns = category_colnames
    for column in categories_2:
       # set each value to be the last character of the string and convert column from string to numeric
        categories_2[column] = categories_2[column].str[-1].astype(int)
        
    categories_2['id'] = categories['id']
    df = messages.merge(categories_2, on='id')
        # drop duplicates
    df.drop_duplicates(keep='first',inplace=True)
    return df

def save_data(df, database_filename):
    """
        Save processed dataframe into sqlite database
    Args: 
        df: The preprocessed dataframe
        database_filename: name of the database
    Returns: 
        None
    """
    # save data into a sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_message', engine, index=False,  if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading and Cleaning data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide 3 arguments in order:'\
              '1. file path for messsages.csv'\
              '2. file path for categories.csv'\
              '3. file path for the database to store cleaned output dataframe'\
              'Example: python ETL_pipeline.py '\
              'data/messages.csv data/categories.csv DisasterResponse.db'
              )


if __name__ == '__main__':
    main()