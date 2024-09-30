"""Data processing module."""
import pandas as pd

from config.config import settings


def label_data(data) -> pd.DataFrame:
    """Function for labeling data.

    This function groups all anomalies
    and assigns a numerical value to
    each set of anomalies to serve as
    class identifiers for the classifier.

    Args:
        data (list):
            List containing a set of data in each item
            reference, mf1, ..., Mfboi.

    Returns:
        DataFrame: A single DataFrame with the
        all the anomalies together.
    """
    return_data = []
    name_datasets = list(settings.path_data.keys())

    for idx in range(len(name_datasets)):
        aux = data[idx]
        aux['Problem_id'] = [idx for jdx in range(len(aux['T2']))]
        return_data.append(aux)

    return_data = pd.concat(return_data)
    return_data = round(return_data, 3).reset_index(drop=True)
    return_data = return_data[settings.variables]

    return return_data
