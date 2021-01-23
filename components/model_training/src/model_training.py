from mlflow.entities.model_registry.model_version_stages import STAGE_PRODUCTION
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
import pandas as pd
import fire
import numpy as np
from singleton_logger import SingletonLogger
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

logger = SingletonLogger.get_logger()
model_name = "xgb_regressor"


def __save(model, name):
    mlflow.xgboost.log_model(model, artifact_path="XGB_regressor_model", registered_model_name=name)
    logger.info("Model saved")


def __log_parameter(name, parameter_value):
    logger.info(name + ": " + str(parameter_value))
    mlflow.log_param(name, parameter_value)


def __log_metric(name, metric_value):
    logger.info(name + ": " + str(metric_value))
    mlflow.log_metric(name, metric_value)


def __save_test_set_to_file(x_test_set, y_test_set):
    x_test_set.to_csv('/tmp/x_test_set.csv')
    y_test_set.to_csv('/tmp/y_test_set.csv')


def __split_data_into_x_y(data):
    target_col = "index"
    x = data.drop(columns=target_col)
    y = data.loc[:, target_col]
    logger.info("Data splitted into x and y")
    return x, y


def __get_mlflow_experiment(name):
    if not mlflow.get_experiment_by_name(name):
        mlflow.create_experiment(name=name)
    return mlflow.get_experiment_by_name(name)


def __promote(name):
    client = MlflowClient()
    client.transition_model_version_stage(name, version=2, stage=STAGE_PRODUCTION)


def train_model(dataset_path, n_estimators, learning_rate, max_depth, min_child_weight, version):
    dataset = pd.read_csv(dataset_path)
    test_set_size = 0.33
    train_set, test_set = train_test_split(dataset, test_size=test_set_size)
    x_training_set, y_training_set = __split_data_into_x_y(train_set)
    x_test_set, y_test_set = __split_data_into_x_y(test_set)

    experiment = __get_mlflow_experiment(model_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=version):
        __log_parameter("N estimators", n_estimators)
        __log_parameter("Learning Rate", learning_rate)
        __log_parameter("Max Depth", max_depth)
        __log_parameter("Min child weight", min_child_weight)
        __log_parameter("Test set size", test_set_size)

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

        logger.info("Cross Validation Started")
        cv_scores = cross_val_score(model, x_training_set, y_training_set, cv=10)

        y_test_pred = model.predict(x_test_set)
        rmse = np.sqrt(mean_squared_error(y_test_pred, y_test_set))
        __log_metric("Mean CV score", cv_scores.mean())
        __log_metric("rmse", rmse)

        __save(model, model_name)

#        __promote(model_name)


if __name__ == "__main__":
    fire.Fire(train_model)
