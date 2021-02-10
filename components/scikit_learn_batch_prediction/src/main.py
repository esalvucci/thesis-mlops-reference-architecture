import fire
import ElectricityConsumptionRegressorService
import pandas as pd


def predict(dataset_path):
    """
    Retreive a model from the MLFlow Model Registry, pack it into a BentoService, and save and register the
    BentoService via BentoML’s built-in model management system.
    """
    bento_service = ElectricityConsumptionRegressorService.load()
    df = pd.read_csv(dataset_path, index_col=0, header=None).iloc[:, :-1]
    df.fillna(-99999, inplace=True)
    prediction = bento_service.predict(df)
    prediction = pd.Series(prediction)
    prediction.to_csv('/tmp/prediction.csv')


if __name__ == "__main__":
    """
    Calls the function used to build the service
    """
    fire.Fire(predict)
