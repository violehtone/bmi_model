#!/usr/bin/env python3
from sklearn.experimental import enable_iterative_imputer # noqa F401 required for imputation to work
from sklearn import linear_model, impute, metrics, model_selection
import pandas as pd
import numpy as np
import pickle
import argparse
import logging
import datetime
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def save_model_on_disk(model: linear_model, output_model_name: str):
    """
    Saves the model on local disk
    """
    logger.info(f"Saving the model as: {output_model_name}...")
    pickle.dump(model, open(output_model_name, 'wb'))
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


def build_model(training_data: str, output_model_name: str,
                imputer: impute.IterativeImputer) -> linear_model:
    """
    Builds a Lasso regression model from the given training data
    """
    preprocessed_data = preprocess_data(data=training_data, imputer=imputer)
    X = preprocessed_data.drop(labels=["BMI", "ID"], axis=1, errors='ignore')
    y = pd.DataFrame(preprocessed_data["BMI"]).values.ravel()

    logger.info("Building a lasso regression model with 10-fold CV")
    cv = model_selection.RepeatedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=1
    )
    model = linear_model.LassoCV(
        alphas=np.arange(0.01, 1, 0.01),
        cv=cv,
        n_jobs=-1,
        tol=0.05
    )
    model.fit(X, y)

    # Save model file on local disk
    save_model_on_disk(model, output_model_name)

    return model


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
    logger.info(mse)

    # Plot the results
    plt.plot(y)
    plt.plot(prediction)
    plt.legend(['Expected', 'Predicted'])
    plt.title('Validation of the model on validation data')
    plt.xlabel('ID')
    plt.ylabel('BMI')
    plt.savefig("plot.png")


def main(training_data: str, test_data: str, output_model_name: str):
    # Define an imputer to be used for the datasets
    imputer = impute.IterativeImputer(max_iter=10, random_state=0, tol=0.01)

    # Build the lasso regression model from the training data
    model = build_model(
        training_data=training_data,
        output_model_name=output_model_name,
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
    parser.add_argument('--output_model_name', type=str, required=True)
    parser.add_argument('--test_data', type=str)
    return vars(parser.parse_args())


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    args = parse_args()
    main(**args)

    # Print the execution time of the script
    execution_time = datetime.datetime.now() - start_time
    logger.info(f"Execution time: {execution_time}")
