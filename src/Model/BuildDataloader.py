"""Build the dataloader for the LightGBM model."""

from typing import Any

import lightgbm as lgb


def build_dataloader(
        x_train,
        x_test,
        x_val,
        y_train,
        y_test,
        y_val,
        ) -> Any:
    """Build the dataloader for the LightGBM.

    Args:
        x_train (_type_): Features used for training
        y_train (_type_): Target used for training
        x_test (_type_): Features used for test
        y_test (_type_): Target used for test
        x_val (_type_): Features used for validation
        y_val (_type_): Target used for validation

    Returns:
        Any: Dataloaders for training, test and validation.
    """
    # Create a LightGBM dataset
    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)
    valid_data = lgb.Dataset(x_val, label=y_val)

    return train_data, test_data, valid_data
