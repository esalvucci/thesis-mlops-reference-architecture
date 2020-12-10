from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys


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


n_estimators = int(sys.argv[1])
learning_rate = float(sys.argv[2])
max_depth = int(sys.argv[3])
min_child_weight = int(sys.argv[4])

model = XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=5
)

model.fit(
    X_train_prep, y_train, early_stopping_rounds=10,
    eval_set=[(X_train_prep, y_train), (X_test_prep, y_test)],
    verbose=False,
)