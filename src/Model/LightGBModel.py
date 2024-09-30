"""Create, predict and evaluate LightGBM."""

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from config.config import settings


def build_model(train_data, test_data):
    """Create the LightGBM Model.

    Args:
        train_data (Dataset): Dataset with training data
        test_data (Dataset): Dataset with the test data

    Returns:
        _type_: Trained model
    """
    return lgb.train(
        settings.params,
        train_data,
        settings.num_round,
        valid_sets=[test_data],
        )


def make_prediction(model, data):
    """Make the prediction of the model.

    Args:
        model (_type_): LightGBM model.
        data (_type_): Data that we want to make a prection.

    Returns:
        _type_: The class with the highest problability.
    """
    y_pred = model.predict(
        data,
        num_iteration=model.best_iteration,
        )

    return np.argmax(y_pred, axis=1)


def check_accuracy(y_test, y_pred):
    """Check the accuracy of the model.

    Args:
        y_test (_type_): Ground truth data
        y_pred (_type_): Data predicted by the model
    """
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')


def score(y_test, y_pred):
    """Make a report about the classification metrics.

    Args:
        y_test (_type_): Ground truth data
        y_pred (_type_): Data predicted by the model
    """
    print(classification_report(y_test, y_pred))
