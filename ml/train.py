import pandas as pd
from ml.preprocess import clean_text

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