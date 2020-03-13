#import statements
import sys
import nltk
nltk.download(['stopwords','punkt', 'wordnet'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from prettytable import PrettyTable
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    
    """
    Load Data Function
    
    """
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("message_disaster", con=engine)
    
    
    # Relevant data for the ML model are loaded.
    # The data from the original message is skipped because we would need to know the language in order 
    # to use a specific lemmatizer and process the text in the original language correctly. 
    # The genre could be added in a second phase if we need to improve the model.

    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    
    """Normalize and tokenize a string in English (because of specific English lemmatizer) from a dataframe, 
    remove stopwords, lemmatize. 
    
    """
    
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # word tokenizer
    words = word_tokenize(text)
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    
    """ Build a pipeline to classify the text into categories
    The second best performing classifier from this project https://medium.com/@robert.salgado/multiclass-text-classification-from-start-to-finish-f616a8642538 - AdaBoost - ist best performing        on this data - This model also can manage imbalanced classed (missing examples of test data for all the categories of the classes) "out of the box" while other classifiers like                LinearSVC are not able to do it, and would require additional over-sampling techniques (such as SMOTE) to creates synthetic samples of the minority classes.

    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
       
        'clf__estimator__n_estimators': [50, 100]
        
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters)    
    return cv

    
    return pipeline


def evaluate_model(model, X_test, Y_test):
    
    """ Test the model and report the f1 score, precision and recall for each output category of the dataset as a pretty table
    """
    
    #predict labels
    y_pred = model.predict(X_test)
    
    # display metrics in table
    t = PrettyTable(["Column", "Precision", "Recall", "F1-Score"])
    for index, column in enumerate(Y_test.columns):
        
        precision,recall,fscore,support=score(Y_test[column].values, y_pred.T[index],average="weighted")
        t.add_row([column, round(precision, 2), round(recall,2), round(fscore,2)])
    print(t)

    
    return


def save_model(model, model_filepath):
    
    """ This function saves trained model as Pickle file, to be loaded later.
    
    """
    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    
    
    return


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
        evaluate_model(model, X_test, Y_test)

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