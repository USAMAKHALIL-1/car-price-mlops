# train.py - trains a car price model and records the run in MLflow
# Usage: python train.py [linear|tree|forest]   (default: linear)

import sys
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Tell MLflow where to store results and which experiment to use
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("car-price-project")

# Choose the model type from the command line
model_type = sys.argv[1] if len(sys.argv) > 1 else "linear"

if model_type == "linear":
    model = LinearRegression()
elif model_type == "tree":
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
elif model_type == "forest":
    model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
else:
    print("Unknown model type. Use: linear, tree or forest")
    sys.exit(1)

# Read the cleaned data
data = pd.read_csv("cars_clean.csv")

# Features (inputs) and label (what we predict)
features = ["year", "cc", "mileage", "imported", "automatic"]
X = data[features]
y = data["price_lakh"]

# Split into training (80%) and testing (20%) sets
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)
print("Cars used for training:", len(X_train))
print("Cars used for testing:", len(X_test))

with mlflow.start_run():

    # Train the model
    model.fit(X_train, y_train)

    # Predict prices for the test cars
    predictions = model.predict(X_test)

    # Show 10 test cars: real price, our guess, and the error
    print("\nyear | cc | mileage | real price | guess | off by")
    for i in range(10):
        row = X_test.iloc[i]
        real = y_test.iloc[i]
        guess = predictions[i]
        print(f"{int(row['year'])} | {int(row['cc'])} | {int(row['mileage'])} | {real} | {guess:.1f} | {abs(real - guess):.1f}")

    # Mean Absolute Error: the average size of the mistakes, in lakh
    mae = mean_absolute_error(y_test, predictions)
    print(f"\nModel: {model_type} | MAE: {mae:.2f} lakh")

    # Log parameters and the metric to MLflow
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("rows_in_dataset", len(data))
    mlflow.log_metric("mae", mae)

    # Save the score to a file so the CI quality gate can read it
    with open("metrics.json", "w") as f:
        json.dump({"mae": mae}, f)

    # Log the model and register it in the MLflow model registry
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="car-price-model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

# Also save the trained model to a file for later use
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")