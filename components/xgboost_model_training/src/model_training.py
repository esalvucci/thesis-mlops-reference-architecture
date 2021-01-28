from mlflow.entities.model_registry.model_version_stages import STAGE_PRODUCTION
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor, DMatrix
from xgboost import cv as xgb_cv
import pandas as pd
import fire
import numpy as np
from singleton_logger import SingletonLogger
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)


logger = SingletonLogger.get_logger()
model_name = "xgb_regressor"
params_grid = dict(
    eta=[0.05, 0.1, 0.3],
    max_depth=[2, 4, 6],
    min_child_weight=[5, 1]
)


def __fit_prep_pipeline(df):
    cat_features = ["month", "weekday", "hour"]  # categorical features
    bool_features = ["holiday"]  # boolean features
    num_features = [c for c in df.columns
                    if c.startswith("load_lag")]  # numerical features
    prep_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_features),
        ("bool", FunctionTransformer(), bool_features),  # identity
        ("num", StandardScaler(), num_features),
    ])
    prep_pipeline = prep_pipeline.fit(df)

    feature_names = []
    one_hot_tf = prep_pipeline.transformers_[0][1]
    for i, cat_feature in enumerate(cat_features):
        categories = one_hot_tf.categories_[i]
        cat_names = [f"{cat_feature}_{c}" for c in categories]
        feature_names += cat_names
    feature_names += (bool_features + num_features)

    return feature_names, prep_pipeline


def __xgb_grid_search_cv(params_grid, X, y, nfold, num_boost_round=1000, early_stopping_rounds=10):
    params_grid = ParameterGrid(params_grid)
    search_results = []
    logger.info(f"Grid search CV : nfold={nfold}, " +
          f"numb_boost_round={num_boost_round}, " +
          f"early_stopping_round={early_stopping_rounds}")
    for params in params_grid:
        logger.info(f"\t{params}")
        cv_df = xgb_cv(
            params=params, dtrain=DMatrix(X, y), nfold=nfold,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            shuffle=False, metrics="rmse",
        )
        cv_results = params.copy()
        cv_results["train-rmse-mean"] = cv_df["train-rmse-mean"].min()
        cv_results["test-rmse-mean"] = cv_df["test-rmse-mean"].min()
        search_results.append(cv_results)
    return pd.DataFrame(search_results)


def __save(model, name):
    mlflow.xgboost.log_model(model, artifact_path="XGB_regressor_model", registered_model_name=name)
    logger.info("Model saved")


def __log_parameter(name, parameter_value):
    logger.info(name + ": " + str(parameter_value))
    mlflow.log_param(name, parameter_value)


def __log_metric(name, metric_value):
    logger.info(name + ": " + str(metric_value))
    mlflow.log_metric(name, metric_value)


def __get_mlflow_experiment(name):
    if not mlflow.get_experiment_by_name(name):
        mlflow.create_experiment(name=name)
    return mlflow.get_experiment_by_name(name)


def __split_data_into_x_y(data):
    target_col = "load"
    x = data.drop(columns=target_col)
    y = data.loc[:, target_col]
    logger.info("Data splitted into x and y")
    return x, y


def __promote(name):
    client = MlflowClient()
    client.transition_model_version_stage(name, version=2, stage=STAGE_PRODUCTION)


def train_model(training_set_path, test_set_path, original_dataset_path,
                n_estimators, learning_rate, max_depth, min_child_weight, version, use_cv):
    training_dataset = pd.read_csv(training_set_path)
    test_dataset = pd.read_csv(test_set_path)
    x_training_set, y_training_set = __split_data_into_x_y(training_dataset)
    x_test_set, y_test_set = __split_data_into_x_y(test_dataset)

    feature_names, prepared_data = __fit_prep_pipeline(x_training_set)
    logger.info(feature_names)
    x_training_set = prepared_data.transform(x_training_set)
    x_training_set = pd.DataFrame(x_training_set, columns=feature_names, index=training_dataset.index)

    x_test_set = prepared_data.transform(x_test_set)
    x_test_set = pd.DataFrame(x_test_set, columns=feature_names, index=test_dataset.index)

    experiment = __get_mlflow_experiment(model_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=version):
        if use_cv:
            xgb_search_scores = __xgb_grid_search_cv(
                params_grid, x_training_set, y_training_set, nfold=4, early_stopping_rounds=10
            )

            parameters_dic = xgb_search_scores.sort_values(by="test-rmse-mean").head(1)
            learning_rate = float(parameters_dic.eta)
            max_depth = int(parameters_dic.max_depth)
            min_child_weight = int(parameters_dic.min_child_weight)

        __log_parameter("N estimators", n_estimators)
        __log_parameter("Learning Rate", learning_rate)
        __log_parameter("Max Depth", max_depth)
        __log_parameter("Min child weight", min_child_weight)

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )

        logger.info("Training Started")
        model.fit(
            x_training_set, y_training_set, early_stopping_rounds=10,
            eval_set=[(x_training_set, y_training_set), (x_test_set, y_test_set)],
            verbose=True,
        )

        y_test_pred = model.predict(x_test_set)
        logger.info("x_test_set")
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        logger.info(x_test_set.head(1).to_csv())
        rmse = np.sqrt(mean_squared_error(y_test_pred, y_test_set))
        __log_metric("rmse", rmse)

        __save(model, model_name)

#        __promote(model_name)


if __name__ == "__main__":
    fire.Fire(train_model)
