"""Module for training, testing and validation."""

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config.config import settings


def train_test_val_split(data):
    """Module for split training, testing and validation.

    This module receives a dataframe with all the
    data together, randomly separates 10% of the data
    for valid_data and separates the other 90% into two sets
    one for training and the other for testing. The separation of the
    two sets is in the proportion defined in DataConfig.
    The resulting data sets are brought to the same
    scale by applying StandarScaler.

    Args:
        data (Dataframe): Dataframe with all the data.

    Returns:
        DataFrame: 6 sets of data for training and validation.
    """
    ss = StandardScaler()

    valid_data = data.sample(
        frac=settings.frac_val,
        random_state=settings.seed,
        )

    x_val = valid_data.drop('Problem_id', axis=1)
    x_val = pd.DataFrame(ss.fit_transform(x_val), columns=x_val.columns)

    y_val = valid_data['Problem_id']

    data = data.drop(valid_data.index)
    data = data.reset_index(drop=True)

    # Shuffle the index of our dataframe
    indices = torch.randperm(len(data)).tolist()

    # Getting 80% of our data to train our model
    train_size = int(settings.frac_train*len(data))
    df_train = data.iloc[indices[:train_size]].reset_index(drop=True)
    x_train = df_train.drop('Problem_id', axis=1)
    x_train = pd.DataFrame(ss.fit_transform(x_train), columns=x_train.columns)

    y_train = df_train['Problem_id']

    # Getting 20% of our data to test our model #0.5
    test_size = int(settings.frac_test*len(data))
    df_test = data.iloc[indices[test_size:]].reset_index(drop=True)
    x_test = df_test.drop('Problem_id', axis=1)
    x_test = pd.DataFrame(ss.fit_transform(x_test), columns=x_test.columns)

    y_test = df_test['Problem_id']

    return x_train, x_test, x_val, y_train, y_test, y_val
