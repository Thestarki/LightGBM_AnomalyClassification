"""Build a confusion matrix."""
from sklearn.metrics import confusion_matrix


def build_matrix(y_test, y_pred):
    """
    Build a confusion matrix.

    Build a confusion matrix to check which
    predition is correct.

    Args:
        y_test (_type_): Ground truth data
        y_pred (_type_): Data predicted by the model
    """
    matrix = confusion_matrix(y_test, y_pred)
    print('Confusion matrix: \n', matrix)
