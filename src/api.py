#!/usr/bin/env python3

from flask import Flask, request
import pickle
import pandas as pd
import json

MODEL_FILENAME = "/app/src/Lasso_model.pkl"
app = Flask(__name__)

# Load the model
model = pickle.load(open(MODEL_FILENAME, 'rb'))


@app.route("/", methods=['GET'])
def ping():
    return "Ping!"


@app.route("/bmi", methods=['POST'])
def predict_bmi():
    # Read data from JSON to dict
    payload = request.get_json(force=True)
    # Convert dict to Pandas df and remove 'BMI' and 'ID' columns
    df = pd.DataFrame.from_dict(payload, orient="index").T
    data = df.drop(labels=["BMI", "ID"], axis=1, errors='ignore')
    # predict BMI with the model
    prediction = model.predict(data)
    # Return prediction as json
    return json.dumps(prediction.tolist())


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
