import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing


def main():
    model = joblib.load("models/linear_regression.pkl")

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop("MedHouseVal", axis=1)

    sample = X.iloc[[0]]

    prediction = model.predict(sample)

    print("Sample Features:\n")
    print(sample)

    print("\nPredicted House Value:")
    print(round(prediction[0], 3))


if __name__ == "__main__":
    main()