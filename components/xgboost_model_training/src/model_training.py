from mlflow.entities.model_registry.model_version_stages import STAGE_PRODUCTION
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import pandas as pd
import fire
import numpy as np
from singleton_logger import SingletonLogger
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)


logger = SingletonLogger.get_logger()
model_name = "random_forest_regressor"
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


def __save(model, name):
    mlflow.sklearn.log_model(model, artifact_path="random_forest_regressor", registered_model_name=name)
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


def train_model(training_set_path, test_set_path, original_dataset_path):
    training_dataset = pd.read_csv(training_set_path)
    test_dataset = pd.read_csv(test_set_path)
    x_training_set, y_training_set = __split_data_into_x_y(training_dataset)
    x_test_set, y_test_set = __split_data_into_x_y(test_dataset)

    feature_names, prepared_data = __fit_prep_pipeline(x_training_set)
    x_training_set = prepared_data.transform(x_training_set)
    x_training_set = pd.DataFrame(x_training_set, columns=feature_names, index=training_dataset.index)

    x_test_set = prepared_data.transform(x_test_set)
    x_test_set = pd.DataFrame(x_test_set, columns=feature_names, index=test_dataset.index)

    experiment = __get_mlflow_experiment(model_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model = RandomForestRegressor(
            n_estimators=100, criterion='mse', min_samples_leaf=0.001, random_state=42
        )

        logger.info("Training Started")
        model.fit(x_training_set, y_training_set)

        y_test_pred = model.predict(x_test_set)
        rmse = np.sqrt(mean_squared_error(y_test_pred, y_test_set))
        __log_metric("rmse", rmse)

        __save(model, model_name)

#        __promote(model_name)


if __name__ == "__main__":
    fire.Fire(train_model)
