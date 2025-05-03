import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def create_corpus(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans text data from the 'Review' column of a DataFrame.
    """
    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus=[]

    for i in range(0, len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

def transform_data(dataset: pd.DataFrame, corpus: list) -> pd.DataFrame:
    """
    Transform the data into a format suitable for machine learning.
    """
    cv = CountVectorizer(max_features = 1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    return X, y

def preprocessing(dataset: pd.DataFrame) -> tuple:
    """
    Preprocess the dataset by creating a corpus and transforming the data.
    """
    corpus = create_corpus(dataset)
    X, y = transform_data(dataset, corpus)

    return X, y