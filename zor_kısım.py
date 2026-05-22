import pandas as pd
import numpy as np

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


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="latin-1")

    # Sadece gerekli 2 kolonu al
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
    print(df.columns.tolist())
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
                max_features=max_features
            )
        ),
        ("model", model)
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

    print("=" * 60)
    print(f"MODEL: {model_label}")
    print("Accuracy:", round(acc, 4))
    print("F1 Score:", round(f1, 4))
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 60)
    print()

    return {
        "name": model_label,
        "pipeline": pipeline,
        "accuracy": acc,
        "f1": f1,
        "y_pred": y_pred,
    }


def show_top_words(best_pipeline: Pipeline, top_n: int = 15) -> None:
    model = best_pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        print("Bu modelde coef_ yok. Top words sadece Logistic Regression için gösteriliyor.")
        print()
        return

    tfidf = best_pipeline.named_steps["tfidf"]
    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_[0]

    top_spam = sorted(zip(coefficients, feature_names), reverse=True)[:top_n]
    top_ham = sorted(zip(coefficients, feature_names))[:top_n]

    print("=== TOP SPAM WORDS ===")
    for coef, word in top_spam:
        print(f"{word}: {round(coef, 3)}")
    print()

    print("=== TOP HAM WORDS ===")
    for coef, word in top_ham:
        print(f"{word}: {round(coef, 3)}")
    print()


def show_wrong_predictions(X_test, y_test, y_pred, max_examples=5) -> None:
    wrong_idx = np.where(y_test.values != y_pred)[0]

    print("=== WRONG PREDICTIONS ===")

    if len(wrong_idx) == 0:
        print("Hiç yanlış tahmin yok.")
        print()
        return

    shown = 0
    for i in wrong_idx:
        real = y_test.iloc[i]
        pred = y_pred[i]

        if real == 0 and pred == 1:
            error_type = "FALSE POSITIVE (Normal -> Spam)"
        elif real == 1 and pred == 0:
            error_type = "FALSE NEGATIVE (Spam -> Normal)"
        else:
            error_type = "UNKNOWN"

        print(error_type)
        print("Mesaj :", X_test.iloc[i])
        print("Gerçek:", real)
        print("Tahmin:", pred)
        print("-" * 60)

        shown += 1
        if shown >= max_examples:
            break

    print()


def show_probabilities(best_pipeline: Pipeline, X_test, y_test, n_examples=5) -> None:
    if not hasattr(best_pipeline.named_steps["model"], "predict_proba"):
        print("Bu model predict_proba desteklemiyor.")
        print()
        return

    probs = best_pipeline.predict_proba(X_test)[:, 1]

    print("=== SAMPLE PROBABILITIES ===")
    for i in range(min(n_examples, len(X_test))):
        print("Mesaj:", X_test.iloc[i])
        print("Gerçek:", y_test.iloc[i])
        print("Spam probability:", round(probs[i], 4))
        print("-" * 60)
    print()


def evaluate_with_threshold(best_pipeline: Pipeline, X_test, y_test, threshold=0.8) -> None:
    if not hasattr(best_pipeline.named_steps["model"], "predict_proba"):
        print("Bu model threshold denemesi için predict_proba desteklemiyor.")
        print()
        return

    probs = best_pipeline.predict_proba(X_test)[:, 1]
    custom_pred = (probs >= threshold).astype(int)

    print(f"=== THRESHOLD EVALUATION (threshold={threshold}) ===")
    print("Accuracy:", round(accuracy_score(y_test, custom_pred), 4))
    print("F1 Score:", round(f1_score(y_test, custom_pred), 4))
    print()
    print("Classification Report:")
    print(classification_report(y_test, custom_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, custom_pred))
    print()


def test_custom_messages(best_pipeline: Pipeline) -> None:
    test_messages = [
        "FREE iPhone now!!!",
        "Hey akşam buluşalım mı",
        "Limited offer click now",
        "Meeting at 5pm",
        "Congratulations you won cash prize",
    ]

    print("=== CUSTOM MESSAGE TEST ===")
    for msg in test_messages:
        pred = best_pipeline.predict([msg])[0]

        if hasattr(best_pipeline.named_steps["model"], "predict_proba"):
            prob = best_pipeline.predict_proba([msg])[0][1]
            print(f"Mesaj: {msg}")
            print(f"Tahmin: {'SPAM' if pred == 1 else 'HAM'}")
            print(f"Spam probability: {round(prob, 4)}")
        else:
            print(f"Mesaj: {msg}")
            print(f"Tahmin: {'SPAM' if pred == 1 else 'HAM'}")

        print("-" * 60)
    print()


def run_cli(best_pipeline: Pipeline) -> None:
    print("=== MINI CLI TOOL ===")
    print("Çıkmak için 'q' yaz.")
    print()

    while True:
        msg = input("Mesaj gir: ").strip()

        if msg.lower() == "q":
            print("Programdan çıkıldı.")
            break

        pred = best_pipeline.predict([msg])[0]

        if hasattr(best_pipeline.named_steps["model"], "predict_proba"):
            prob = best_pipeline.predict_proba([msg])[0][1]
            print(f"Sonuç: {'SPAM' if pred == 1 else 'HAM'} | Spam probability: {round(prob, 4)}")
        else:
            print(f"Sonuç: {'SPAM' if pred == 1 else 'HAM'}")

        print()


def main():
    # 1) Veri yükle
    file_path = "spam.csv"
    df = load_and_prepare_data(file_path)

    # 2) Veri incele
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

    print("Train size:", len(X_train))
    print("Test size :", len(X_test))
    print()

    # 5) Farklı deneyler
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
            max_features=max_features
        )

        result = evaluate_model(
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_label=exp_name
        )

        results.append(result)

    # 6) En iyi modeli seç
    best_result = max(results, key=lambda x: x["f1"])
    best_pipeline = best_result["pipeline"]

    print("=" * 60)
    print("EN IYI MODEL")
    print("Model   :", best_result["name"])
    print("Accuracy:", round(best_result["accuracy"], 4))
    print("F1 Score:", round(best_result["f1"], 4))
    print("=" * 60)
    print()

    # 7) Modelin öğrendiği kelimeler
    show_top_words(best_pipeline, top_n=15)

    # 8) Yanlış tahminleri incele
    show_wrong_predictions(
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        y_pred=best_result["y_pred"],
        max_examples=5
    )

    # 9) Probability göster
    show_probabilities(
        best_pipeline=best_pipeline,
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        n_examples=5
    )

    # 10) Threshold dene
    evaluate_with_threshold(
        best_pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test,
        threshold=0.8
    )

    # 11) Kendi test mesajlarını dene
    test_custom_messages(best_pipeline)

    # 12) CLI tool
    run_cli(best_pipeline)


if __name__ == "__main__":
    main()