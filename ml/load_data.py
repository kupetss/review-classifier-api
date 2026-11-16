from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data")


def load_imdb():
    ds = load_dataset("stanfordnlp/imdb")

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    DATA_DIR.mkdir(exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print("train:", train_df.shape)
    print("test:", test_df.shape)


if __name__ == "__main__":
    load_imdb()