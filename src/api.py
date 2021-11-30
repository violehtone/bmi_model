#!/usr/bin/env python3
import pickle
import pandas as pd
import json
from flask import Flask, request

app = Flask(__name__)

# Load the lasso regression model
with open("src/Lasso_model.pkl", "rb") as model:
    LASSO_MODEL = pickle.load(model)
# Load the list of required features for prediction
with open("utils/required_features.txt", "rb") as features:
    REQUIRED_FEATURES = pickle.load(features)


@app.route("/ping", methods=['GET'])
def ping():
    """Ping the server to verify that it is working."""
    return "Ping!"


@app.route("/bmi", methods=['POST'])
def predict_bmi():
    """Predict a person's BMI from a JSON"""

    # Read data from JSON and convert to properly formatted pandas dataframe
    payload = request.get_json(force=True)
    df = pd.DataFrame.from_dict(payload, orient="index").T

    # Return a bad request (400) if the request body don't have all the required features to make the prediction
    if not all(feature in list(df.columns.values) for feature in REQUIRED_FEATURES):
        missing_features = list(set(REQUIRED_FEATURES) - set(list(df.columns.values)))
        return f"One (or more) of the required features are missing. The missing features are: {missing_features}", 400

    # Predict BMI with the model
    prediction = LASSO_MODEL.predict(df)
    return json.dumps({"bmi": prediction[0]}), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
