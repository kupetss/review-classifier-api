import pandas as pd
from ml.preprocess import clean_text
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def train_model(train_df):
    model = make_pipeline(
        TfidfVectorizer(max_features=50000, ngram_range=(1,2)),
        LogisticRegression(max_iter=1000),
    )
    model.fit(train_df["text"], train_df["label"])
    
    return model


def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    train_df = train_df.drop_duplicates(subset="text")
    train_df = train_df.sample(frac=1, random_state=52)

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_data()
    
    print("train:", train_df.shape)
    print("test:", test_df.shape)
    
    model = train_model(train_df)

    predcts = model.predict(test_df["text"])
    accuracy = accuracy_score(test_df["label"], predcts)
    print(f"accuracy:{accuracy:.3f}")