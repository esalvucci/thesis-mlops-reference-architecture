import pandas as pd
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class ElectricityConsumptionRegressorService(BentoService):
    """
    A minimum prediction service exposing a XGBoost model
    """

    @api(input=DataframeInput(), batch=True)
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        data = pd.DataFrame(df)
        return self.artifacts.model.predict(data)
