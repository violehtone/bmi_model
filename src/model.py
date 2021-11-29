#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import logging
import datetime
from sklearn.experimental import enable_iterative_imputer # noqa F401 required for imputation to work
from sklearn import linear_model, impute, metrics, model_selection

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Output model name
MODEL_FILENAME = "src/Lasso_model.pkl"


def save_model_on_disk(model: linear_model):
    """
    Saves the model on local disk
    """
    logger.info(f"Saving the model as: {MODEL_FILENAME}...")
    pickle.dump(model, open(MODEL_FILENAME, 'wb'))
    logger.info("Done!")


def preprocess_data(data: str, imputer: impute.IterativeImputer) \
        -> pd.DataFrame:
    """
    Reads the input data (.tsv file) into a Pandas dataframe and performs
    imputation in case of missing values
    """
    logger.info(f"Reading {data} into a dataframe...")
    training_data = pd.read_csv(data, sep='\t')

    logger.info(f"Imputing missing values (if any) in the {data}...")
    return pd.DataFrame(
        data=imputer.fit_transform(training_data),
        columns=training_data.columns
    )


def build_model(training_data: str, imputer: impute.IterativeImputer) \
        -> linear_model:
    """
    Builds a Lasso regression model from the given training data
    """
    # Preprocess the training data
    preprocessed_data = preprocess_data(data=training_data, imputer=imputer)
    X = preprocessed_data.drop(labels=["BMI", "ID"], axis=1, errors='ignore')
    y = pd.DataFrame(preprocessed_data["BMI"]).values.ravel()

    logger.info("Building a lasso regression model with 10-fold CV")
    # Define the cross validation strategy
    cv = model_selection.RepeatedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=1
    )
    # Define a Lasso linear model with CV
    model = linear_model.LassoCV(
        alphas=np.arange(0.01, 1, 0.01),
        cv=cv,
        n_jobs=-1,
        tol=0.05
    )
    # Fit the model
    model.fit(X, y)

    # Save model file on local disk
    save_model_on_disk(model=model)
    return model


def create_plot(expected: np.ndarray, prediction: np.ndarray, mse: float):
    """
    Create a simple plot of the expected vs. predicted BMI values and save
    the file as 'plot.png'
    """
    plt.plot(expected)
    plt.plot(prediction)
    plt.legend(['Expected', 'Predicted'])
    plt.title(f'Validation of the model on validation data\nMSE={mse}')
    plt.xlabel('ID')
    plt.ylabel('BMI')
    plt.savefig("plot.png")


def validate_model(test_data: str, imputer: impute.IterativeImputer,
                   model: linear_model) -> None:
    """
    Validate the built model with a test data set.
    :param test_data: .tsv file
    :param imputer: imputer to replace missing values
    :param model: linear model
    """
    logger.info(f"Validating the model with {test_data}...")
    preprocessed_test_data = preprocess_data(data=test_data, imputer=imputer)
    X = preprocessed_test_data.drop(
        labels=["BMI", "ID"],
        axis=1,
        errors='ignore'
    )
    y = pd.DataFrame(preprocessed_test_data["BMI"]).values.ravel()
    prediction = model.predict(X)

    # Calculate accuracy of the predictions
    mse = metrics.mean_squared_error(
        y.tolist(),
        prediction
    )
    logger.info(f"Mean squared error (MSE) of the model: {str(mse)}")

    # Plot the results
    create_plot(expected=y, prediction=prediction, mse=mse)


def main(training_data: str, test_data: str):
    # Define a multivariate imputer to be used for the datasets
    imputer = impute.IterativeImputer(max_iter=10, random_state=0, tol=0.01)

    # Build the lasso regression model from the training data
    model = build_model(
        training_data=training_data,
        imputer=imputer
    )

    # Validate the model with the test data set
    if test_data:
        validate_model(
            test_data=test_data,
            imputer=imputer,
            model=model
        )


def parse_args():
    """
    Parses the input arguments. This model building script requires two .tsv
    files as its input. One for training the lasso regression model and
    another one for optimizing it.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, required=True)
    parser.add_argument('--test_data', type=str)
    return vars(parser.parse_args())


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    args = parse_args()
    main(**args)

    # Print the execution time of the script
    execution_time = datetime.datetime.now() - start_time
    logger.info(f"Model building execution time: {execution_time}")
