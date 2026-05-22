import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("Çalışma klasörü:", os.getcwd())
print("Dosyalar:", os.listdir())

# dataset yükle
df = pd.read_csv("spam.csv", encoding="latin-1")

# gereksiz kolonları at
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# label encode
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# head
print(df.head())

# shape
print(df.shape)

# columns
print(df.columns)

# missing values
print(df.isnull().sum())

# duplicate
print(df.duplicated().sum())

# label dağılımı
print(df["label"].value_counts())

X = df["message"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Tahminler hazır.")
print(y_pred[:10])
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))