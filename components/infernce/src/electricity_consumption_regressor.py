import pandas as pd

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.xgboost import XgboostModelArtifact
from xgboost import DMatrix
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


@env(infer_pip_packages=True)
@artifacts([XgboostModelArtifact('model')])
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
        features = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
                    'month_10', 'month_11', 'month_12', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4',
                    'weekday_5', 'weekday_6', 'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6',
                    'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15',
                    'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'holiday',
                    'load_lag_24', 'load_lag_25', 'load_lag_26', 'load_lag_27', 'load_lag_28', 'load_lag_29',
                    'load_lag_30', 'load_lag_31', 'load_lag_32', 'load_lag_33', 'load_lag_34', 'load_lag_35',
                    'load_lag_36', 'load_lag_37', 'load_lag_38', 'load_lag_39', 'load_lag_40', 'load_lag_41',
                    'load_lag_42', 'load_lag_43', 'load_lag_44', 'load_lag_45', 'load_lag_46', 'load_lag_47',
                    'load_lag_48']
        data = pd.DataFrame(df, columns=features)
        return self.artifacts.model.predict(DMatrix(data))
