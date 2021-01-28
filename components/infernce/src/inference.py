import fire
from singleton_logger import SingletonLogger
from electricity_consumption_regressor import ElectricityConsumptionRegressorService
import mlflow
logger = SingletonLogger.get_logger()


def __get_data(model_path):
    model = mlflow.xgboost.load_model(model_path)
    electricity_consumption_regressor_service = ElectricityConsumptionRegressorService()
    electricity_consumption_regressor_service.pack('model', model)
    electricity_consumption_regressor_service.save()


if __name__ == "__main__":
    fire.Fire(__get_data)
