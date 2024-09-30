"""This module runs everything in the LIghtGBM folder."""

import sys

from Model.BuildConfusionMatrix import build_matrix
from Model.BuildDataloader import build_dataloader
from Model.LightGBModel import (build_model, check_accuracy, make_prediction,
                                score)
from Model.ProcessData import label_data
from Model.ReadData import read_data
from Model.TrainTestValSplit import train_test_val_split

sys.path.append('LightGBM')

data = read_data()

data = label_data(data)

x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(data)

train_loader, test_loader, val_loader = build_dataloader(
    x_train, x_test, x_val, y_train, y_test, y_val,
)

model = build_model(train_loader, test_loader)

y_pred = make_prediction(model, x_test)
check_accuracy(y_test, y_pred)
build_matrix(y_test, y_pred)
score(y_test, y_pred)

y_pred_val = make_prediction(model, x_val)
check_accuracy(y_val, y_pred_val)
build_matrix(y_val, y_pred_val)
score(y_val, y_pred_val)
