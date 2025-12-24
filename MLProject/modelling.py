import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

def main(data_path):
    df = pd.read_csv(data_path)

    target_col = "target"
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # AUTLOG
    mlflow.sklearn.autolog()

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Accuracy: {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)