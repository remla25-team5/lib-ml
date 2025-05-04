


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from lib_ml.preprocessing import create_corpus
import numpy as np


class RestaurantSentimentPreprocessor(BaseEstimator, TransformerMixin):
    """
    A wrapper for CountVectorizer that processes text data from a pandas DataFrame.
    This class is designed to be used in a scikit-learn pipeline.
    """
    def __init__(self, max_features=1420):
        self.max_features = max_features
        self.cv_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learns the CountVectorizer vocabulary using create_corpus.
        X: pandas DataFrame containing the review_column.
        y: Ignored.
        """
        print("Fitting RestaurantSentimentPreprocessor...")

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input X must be a pandas DataFrame, got {type(X)}")
        if 'Review' not in X.columns:
            raise ValueError("Input DataFrame X must contain the column 'Review'")

        corpus = create_corpus(X) 

        cv = CountVectorizer(max_features=self.max_features)
        self.cv_ = cv.fit(corpus) 
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms text data using create_corpus and the fitted CountVectorizer.
        X: pandas DataFrame containing the review_column.
        Returns: np.ndarray (Document-term matrix)
        """
        print("Transforming with RestaurantSentimentPreprocessor...")
        if self.cv_ is None:
            raise RuntimeError("This RestaurantSentimentPreprocessor instance is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
             raise TypeError(f"Input X must be a pandas DataFrame, got {type(X)}")
        if 'Review' not in X.columns:
            raise ValueError("Input DataFrame X must contain the column 'Review'")

        corpus = create_corpus(X)

        X_transformed = self.cv_.transform(corpus).toarray()
        return X_transformed