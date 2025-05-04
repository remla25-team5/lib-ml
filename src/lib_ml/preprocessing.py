import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def clean_text(text: str) -> str:
    """
    Cleans the input text by.
    """
    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

def preprocess_dataset(dataset: pd.DataFrame) -> tuple:
    """
    Preprocesses the dataset by cleaning the text and separating features and labels.
    """
    corpus=[]

    for i in range(0, len(dataset)):
        corpus.append(clean_text(dataset['Review'][i]))
    X = corpus

    y = None
    if dataset.shape[1] > 1:
        y = dataset.iloc[:, 1].values
    return X, y

def preprocess_element(text: str) -> str:
    """
    Preprocesses a single text element by cleaning it.
    """
    return clean_text(text)