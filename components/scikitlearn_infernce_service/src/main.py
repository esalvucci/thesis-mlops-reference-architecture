import fire
from utility.singleton_logger import SingletonLogger
from regressor_service import ElectricityConsumptionRegressorService
import mlflow

logger = SingletonLogger.get_logger()


def build_service(model_path):
    """
    Retreive a model from the MLFlow Model Registry, pack it into a BentoService, and save and register the
    BentoService via BentoML’s built-in model management system.
    """
    model = mlflow.sklearn.load_model(model_path)
    electricity_consumption_regressor_service = ElectricityConsumptionRegressorService()
    electricity_consumption_regressor_service.pack('model', model)
    electricity_consumption_regressor_service.save()


if __name__ == "__main__":
    """
    Calls the function used to build the service
    """
    fire.Fire(build_service)
