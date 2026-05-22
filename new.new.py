import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin-1")

    # İlk 2 kolonu al
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

    # Label encode
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("=== HEAD ===")
    print(df.head())
    print()

    print("=== SHAPE ===")
    print(df.shape)
    print()

    print("=== COLUMNS ===")
    print(df.columns)
    print()

    print("=== MISSING VALUES ===")
    print(df.isnull().sum())
    print()

    print("=== DUPLICATE COUNT ===")
    print(df.duplicated().sum())
    print()

    print("=== LABEL DISTRIBUTION ===")
    print(df["label"].value_counts())
    print()


def build_pipeline(model_name: str, ngram_range=(1, 1), max_features=3000) -> Pipeline:
    if model_name == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "naive_bayes":
        model = MultinomialNB()
    else:
        raise ValueError(f"Bilinmeyen model adı: {model_name}")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=ngram_range,
                max_features=max_features,
            ),
        ),
        ("model", model),
    ])

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    model_label: str,
) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"=== {model_label} ===")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

    return {
        "name": model_label,
        "pipeline": pipeline,
        "accuracy": acc,
        "f1": f1,
        "y_pred": y_pred,
    }


def show_wrong_predictions(X_test, y_test, y_pred, max_examples=5) -> None:
    wrong_idx = np.where(y_test.values != y_pred)[0]

    print("=== WRONG PREDICTIONS ===")
    if len(wrong_idx) == 0:
        print("Hiç yanlış tahmin yok.")
        return

    for i in wrong_idx[:max_examples]:
        print("Mesaj:", X_test.iloc[i])
        print("Gerçek:", y_test.iloc[i])
        print("Tahmin:", y_pred[i])
        print("-" * 50)


def main():
    # 1) Veri yükle
    file_path = "spam.csv"   # gerekirse yolunu değiştir
    df = load_and_prepare_data(file_path)

    # 2) Veri inceleme
    inspect_data(df)

    # 3) Feature / label
    X = df["message"]
    y = df["label_num"]

    # 4) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 5) Farklı model + TF-IDF kombinasyonları
    experiments = [
        ("Logistic (1,1)", "logistic", (1, 1), 3000),
        ("Logistic (1,2)", "logistic", (1, 2), 5000),
        ("NaiveBayes (1,1)", "naive_bayes", (1, 1), 3000),
        ("NaiveBayes (1,2)", "naive_bayes", (1, 2), 5000),
    ]

    results = []

    for exp_name, model_name, ngram_range, max_features in experiments:
        pipeline = build_pipeline(
            model_name=model_name,
            ngram_range=ngram_range,
            max_features=max_features,
        )

        result = evaluate_model(
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_label=exp_name,
        )

        results.append(result)

    # 6) En iyi modeli seç
    best_result = max(results, key=lambda x: x["f1"])

    print("\n" + "=" * 60)
    print("EN IYI MODEL:")
    print("Model:", best_result["name"])
    print("Accuracy:", round(best_result["accuracy"], 4))
    print("F1 Score:", round(best_result["f1"], 4))
    print("=" * 60)
    print()

    # 7) Yanlış tahminleri incele
    show_wrong_predictions(
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        y_pred=best_result["y_pred"],
        max_examples=5,
    )


if __name__ == "__main__":
    main()