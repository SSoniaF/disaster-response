import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load the messages and the categories .csv files into 2 dataframes and merge the 2 dataframes using the common id
       
    """ 
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, how='left', on="id")
  
    return df
    

def clean_data(df):
    
    """ Clean the dataframe:
    - split the values in the categories column on the ; character so that each value becomes a separate column
    - rename the columns with the name of the category
    - convert category values to just numbers 0 or 1
    - replace category column in df with new categories columns
    - remove duplicates from dataframe
    
    Output: clean dataframe
       
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = list(categories.iloc[0, :])
    
    #extract the name of the columns from the first row and add them to a list
    category_colnames = []
    for word in row:
        word = word[:-2]
        category_colnames.append(word)
        
    # rename the columns of `categories` dataframe
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1])) # convert column from string to numeric
    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep='last',inplace=True) 
    
    return df


def save_data(df, database_filename):
    """ Save the clean dataset into an sqlite database, if the file already exists it replaces it
    
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('message_disaster', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()