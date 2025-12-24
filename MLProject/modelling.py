import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import os

def main(data_path):
    df = pd.read_csv(data_path)
    target_col = "target"  # sesuaikan dengan kolom target di dataset
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Accuracy: {score}")

        mlflow.log_metric("accuracy", score)
        mlflow.sklearn.log_model(model, "model")

        # Save model ke folder artifacts
        model_dir = "artifacts"
        os.makedirs(model_dir, exist_ok=True)
        mlflow.sklearn.save_model(model, os.path.join(model_dir, "model"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)
