from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import fire


def compute_learning_curves(model, X, y, curve_step, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    n_train_obs = X_train.shape[0]

    ###                                                 ###
    #    n_iter = math.ceil(n_train_obs / curve_step)     #
    ###                                                 ###
    n_iter = 1
    train_errors, val_errors, steps = [], [], []
    for i in range(n_iter):
        n_obs = (i + 1) * curve_step
        n_obs = min(n_obs, n_train_obs)
        model.fit(X_train[:n_obs], y_train[:n_obs])

        y_train_predict = model.predict(X_train[:n_obs])
        y_val_predict = model.predict(X_val)

        train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
        val_mse = mean_squared_error(y_val, y_val_predict)

        train_errors.append(train_mse)
        val_errors.append(val_mse)
        steps.append(n_obs)
        if verbose:
            msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                i + 1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
            )
            print(msg)
    return steps, train_errors, val_errors


def train_model(x_training_set_path, y_training_set_path, x_test_set_path, y_test_set_path):
    n_estimators = 1000
    learning_rate = 0.5
    max_depth = 5

    x_training_set = pd.read_csv(x_training_set_path, index_col='time')
    y_training_set = pd.read_csv(y_training_set_path, index_col='time')
    x_test_set = pd.read_csv(x_test_set_path, index_col='time')
    y_test_set = pd.read_csv(y_test_set_path, index_col='time')

    print(x_training_set.dtypes)

    model = XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=5
    )

    model.fit(
        x_training_set, y_training_set, early_stopping_rounds=10,
        eval_set=[(x_training_set, y_training_set), (x_test_set, y_test_set)],
        verbose=False,
    )

    pickle.dump(model, open('/tmp/trained_model.pkl', 'wb'))


if __name__ == "__main__":
    fire.Fire(train_model)
