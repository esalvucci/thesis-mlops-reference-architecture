import fire
from singleton_logger import SingletonLogger
from electricity_consumption_regressor import ElectricityConsumptionRegressorService
import mlflow

logger = SingletonLogger.get_logger()


def __get_mlflow_experiment(name):
    if not mlflow.get_experiment_by_name(name):
        mlflow.create_experiment(name=name)
    return mlflow.get_experiment_by_name(name)


def __get_data():
    model_name = "random_forest_regressor"
    stage = "Production"
    experiment = __get_mlflow_experiment("Default")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
        electricity_consumption_regressor_service = ElectricityConsumptionRegressorService()
        electricity_consumption_regressor_service.pack('model', model)
        electricity_consumption_regressor_service.save()


if __name__ == "__main__":
    fire.Fire(__get_data)
