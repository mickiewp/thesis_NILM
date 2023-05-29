import os

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from tqdm import tqdm
from utils import prepare_data
from metrics import regression_metrics, classification_metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

regression_models = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor(), SVR()]
classification_models = [LogisticRegression(max_iter=1000), RandomForestClassifier(), GradientBoostingClassifier(),
                         SVC(probability=True)]
time_windows = [5, 10, 15, 20]
# time_windows = [10]
num_folds = 5

PATH_DATA = "data"
PATH_OUTPUT = os.path.join(PATH_DATA, "output")
os.makedirs(PATH_OUTPUT, exist_ok=True)

# 1 mains
# 2 fridge
# 3 electric oven
# 4 kettle
# 5 stove
path_csv = os.path.join(PATH_DATA, "{}.csv")
path_mains_csv = path_csv.format("1")
devices_dict = {2: "fridge", 3: "electric_oven", 4: "kettle", 5: "stove"}


def prepare_regression_plot(y_true, y_pred, device_name, model_name, time_window):
    fig, ax = plt.subplots()
    ax.plot(y_true, label='True Values')
    ax.plot(y_pred, label='Predicted Values')
    ax.set_xlabel('timestamp')
    ax.set_ylabel('W')
    ax.set_title(f'W regression for {device_name} with {model_name} and time window: {time_window}')
    ax.legend()
    return fig


def save_regression_visualizations(path_device_folder, device, model_name, time_window, y_tests, y_preds):
    y_test_flatt = [prediction for fold_y_test in y_tests for prediction in fold_y_test]
    y_pred_flatt = [prediction for fold_y_pred in y_preds for prediction in fold_y_pred]

    fig_name = f"REG_{device}_{model_name}_{time_window}.png"
    path_fig = os.path.join(path_device_folder, fig_name)
    fig = prepare_regression_plot(y_test_flatt, y_pred_flatt, device, model_name, time_window)
    fig.savefig(path_fig)


def save_classification_visualizations(path_device_folder, device, model_name, time_window, y_tests, y_preds,
                                       y_preds_proba):
    pass


def save_preds(path_device_folder, device_name, model_name, time_window, y_tests, y_preds, regression=True):
    path_file = os.path.join(path_device_folder, f"regression_preds_{time_window}.csv") if regression else os.path.join(
        path_device_folder, f"classification_preds_{time_window}.csv")
    y_test_flatt = [prediction for fold_y_test in y_tests for prediction in fold_y_test]
    y_pred_flatt = [prediction for fold_y_pred in y_preds for prediction in fold_y_pred]
    if not regression:
        y_pred_flatt = np.array(y_pred_flatt)[:, 1]
    exp_name = f"REG_{device_name}_{model_name}_{time_window}"
    if os.path.exists(path_file):
        df = pd.read_csv(path_file, index_col=0)
        df[exp_name] = y_pred_flatt
        df.to_csv(path_file)
    else:
        predictions_dict = {"y_true": y_test_flatt, exp_name: y_pred_flatt}
        df = pd.DataFrame(predictions_dict)
        df.to_csv(path_file)


def save_results(path, results):
    if os.path.exists(path):
        results.to_csv(path, mode='a', header=False)
    else:
        results.to_csv(path)


def evaluate_regression_results(y_tests, y_preds, model, time_window):
    results_dict = {}
    results_dict["model"] = []
    results_dict["time_window"] = []
    results_dict["fold"] = []
    for metric_name in regression_metrics.keys():
        results_dict[metric_name] = []

    for fold, (y_test, y_pred) in enumerate(zip(y_tests, y_preds)):
        results_dict["fold"].append(fold)
        results_dict["model"].append(model)
        results_dict["time_window"].append(time_window)
        for metric_name, metric_func in regression_metrics.items():
            results_dict[metric_name].append(metric_func(y_test, y_pred))

    return pd.DataFrame(results_dict)


def evaluate_classification_results(y_tests, y_preds, y_preds_proba, model, time_window):
    results_dict = {}
    results_dict["model"] = []
    results_dict["time_window"] = []
    results_dict["fold"] = []
    results_dict["TP"] = []
    results_dict["TN"] = []
    results_dict["FP"] = []
    results_dict["FN"] = []
    results_dict["AUC"] = []
    for metric_name in classification_metrics.keys():
        results_dict[metric_name] = []

    for fold, (y_test, y_pred, y_pred_proba) in enumerate(zip(y_tests, y_preds, y_preds_proba)):
        results_dict["fold"].append(fold)
        results_dict["model"].append(model)
        results_dict["time_window"].append(time_window)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        results_dict["TP"].append(tp)
        results_dict["TN"].append(tn)
        results_dict["FP"].append(fp)
        results_dict["FN"].append(fn)
        try:
            results_dict["AUC"].append(roc_auc_score(y_test, y_pred_proba[:, 1]))
        except:
            results_dict["AUC"].append(1.0)
        for metric_name, metric_func in classification_metrics.items():
            results_dict[metric_name].append(metric_func(y_test, y_pred))

    return pd.DataFrame(results_dict)


def run_model_training(X, y, model, regression=True):
    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(X)

    y_tests = []
    y_preds = []
    y_preds_proba = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        x_train = X[train_index]
        y_train = y[train_index]
        x_test = X[test_index]
        y_test = y[test_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred[y_pred < 0] = 0
        y_pred_proba = model.predict_proba(x_test) if not regression else None
        y_tests.append(y_test)
        y_preds.append(y_pred)
        y_preds_proba.append(y_pred_proba)

    return y_tests, y_preds, y_preds_proba


def run_training_for_one_device(device_id, device_name):
    path_device_folder = os.path.join(PATH_OUTPUT, device_name)
    path_device_csv = path_csv.format(device_id)
    path_regression_results = os.path.join(path_device_folder, "regression_results.csv")
    path_classification_results = os.path.join(path_device_folder, "classification_results.csv")
    os.makedirs(path_device_folder, exist_ok=True)
    print(f"Starting training for {device_name}")

    for time_window in tqdm(time_windows):
        print(f"time window: {time_window}")
        X, y = prepare_data(main_file=path_mains_csv, target_file=path_device_csv, time_window=time_window)
        X = np.array(X)
        y = np.array(y)
        y_binary = y.copy()
        y_binary[y_binary > 0] = 1
        y_binary[y_binary <= 0] = 0
        for reg_model in regression_models:
            model_name = reg_model.__class__.__name__
            print(model_name)
            y_tests, y_preds, _ = run_model_training(X, y, reg_model, regression=True)
            results = evaluate_regression_results(y_tests, y_preds, model_name, time_window)
            save_results(path_regression_results, results)
            save_regression_visualizations(path_device_folder, device_name, model_name, time_window, y_tests, y_preds)
            save_preds(path_device_folder, device_name, model_name, time_window, y_tests, y_preds)

        for cls_model in classification_models:
            model_name = cls_model.__class__.__name__
            print(model_name)
            y_tests, y_preds, y_preds_proba = run_model_training(X, y_binary, cls_model, regression=False)
            results = evaluate_classification_results(y_tests, y_preds, y_preds_proba, model_name, time_window)
            save_results(path_classification_results, results)
            save_classification_visualizations(path_device_folder, device_name, model_name, time_window, y_tests,
                                               y_preds,
                                               y_preds_proba)
            save_preds(path_device_folder, device_name, model_name, time_window, y_tests, y_preds_proba,
                       regression=False)


def train():
    print("Starting training...")
    for device_id, device_name in tqdm(devices_dict.items()):
        run_training_for_one_device(device_id, device_name)


if __name__ == '__main__':
    train()
