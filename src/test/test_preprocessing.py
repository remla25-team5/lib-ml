import pandas as pd

from lib_ml.preprocessing import clean_text, preprocess_dataset

# Test cases for Feature and Data 7: All input feature code is tested

def test_clean_text_basic():
    text = "This is a simple test! Not bad at all."
    cleaned = clean_text(text)
    assert "not" in cleaned
    assert "is" not in cleaned
    assert all(word.islower() for word in cleaned.split())
    assert not any(char in cleaned for char in "!@#$%^&*()")

def test_clean_text_empty():
    text = ""
    cleaned = clean_text(text)
    assert cleaned == ""

def test_clean_text_only_stopwords():
    text = "the and is in at"
    cleaned = clean_text(text)
    assert cleaned == ""

def test_preprocess_dataset_basic():
    data = {
        "Review": [
            "I loved the restaurant! Not bad at all.",
            "Terrible restaurant, would not recommend.",
        ],
        "Sentiment": [1, 0]
    }
    df = pd.DataFrame(data)
    X, y = preprocess_dataset(df)
    assert len(X) == 2
    assert isinstance(X[0], str)
    assert list(y) == [1, 0]

def test_preprocess_dataset_no_label():
    data = {
        "Review": [
            "Great restaurant.",
            "Not good."
        ]
    }
    df = pd.DataFrame(data)
    X, y = preprocess_dataset(df)
    assert len(X) == 2
    assert y is None
