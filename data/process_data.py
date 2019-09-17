import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

#Setting pandas display options to easier review printed out dataframes
pd.set_option("display.max_rows",152)
pd.set_option("display.max_columns",None)
pd.set_option("display.max_colwidth", -1)
pd.set_option("display.width",None)

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on="id")
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";",expand=True)

    # selecting the first row of the categories dataframe
    row = categories.iloc[0]

    # using this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

    # convert category values to just numbers 0 or 1
    # Iterate through the category columns in df to keep only the last character
    # of each string (the 1 or 0). For example, related-0 becomes 0, related-1
    # becomes 1. Converting the string to a numeric value.

    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[len(x)-1:])

        # converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replacing categories column in df with new category columns.
    df = df.drop("categories", axis=1)
    df.head()

    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)

    df = pd.concat([df,categories],sort = False,axis=1)

    # Removing duplicates
    # checking number of duplicates
    duplicatedrows = df[df.duplicated(['message'])]
    print(len(duplicatedrows))

    # drop duplicates
    df = df.drop_duplicates(['message'])

    # number of duplicates
    duplicatedrows = df[df.duplicated(['message'])]
    print(len(duplicatedrows))
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # Extracting tablename from database_filename path
    table_name = extract_table_name(database_filename)
    df.to_sql(table_name, engine, index=False)

def extract_table_name(database_filename):
    """Extracts a name of a table from a given database_filename
    """

    table_name = re.search("([^/]+)(\.db)",database_filename)
    return table_name[1]

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
