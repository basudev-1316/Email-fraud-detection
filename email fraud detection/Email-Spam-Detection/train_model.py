import pickle
from pathlib import Path
import string
import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

def transform_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # use regex tokenizer to avoid NLTK punkt dependency
    tokens = re.findall(r"\w+", text)
    words = [w for w in tokens if w.isalnum()]
    words = [w for w in words if w not in stopwords.words("english")]
    stems = [ps.stem(w) for w in words]
    return " ".join(stems)

def main():
    base = Path(__file__).parent
    csv_path = base / "spam.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin-1")
    # drop extra unnamed columns if present
    for col in ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    df.rename(columns={"v1": "target", "v2": "text"}, inplace=True)
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["target"])  # ham=0, spam=1
    df.drop_duplicates(inplace=True)
    df["new_text"] = df["text"].apply(transform_text)

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["new_text"])
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    rf = RandomForestClassifier(n_estimators=100, random_state=2, n_jobs=-1)
    rf.fit(X_train, y_train)

    # save artifacts
    with open(base / "vectorizer.pkl", "wb") as vf:
        pickle.dump(tfidf, vf, protocol=pickle.HIGHEST_PROTOCOL)
    with open(base / "model.pkl", "wb") as mf:
        pickle.dump(rf, mf, protocol=pickle.HIGHEST_PROTOCOL)

    print("Training complete. Saved: vectorizer.pkl, model.pkl")

if __name__ == "__main__":
    main()