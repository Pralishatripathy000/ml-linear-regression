import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
        "Value": [mae, mse, rmse, r2]
    })

    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    metrics.to_csv("outputs/metrics.csv", index=False)
    coefficients.to_csv("outputs/coefficients.csv", index=False)

    joblib.dump(model, "models/linear_regression.pkl")

    print(metrics)


if __name__ == "__main__":
    main()