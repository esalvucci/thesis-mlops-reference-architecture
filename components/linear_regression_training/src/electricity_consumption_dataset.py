from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)
from utility.singleton_logger import SingletonLogger
import pandas as pd

logger = SingletonLogger.get_logger()


class ElectricityConsumptionDataset:
    """
    This class encapsulates the electricity consumption dataset and let to perform some data preparation steps on it
    """
    def __init__(self, df):
        self.df = df
        self.training_set, self.test_set = train_test_split(self.df, test_size=0.33)
        self.feature_names = []
        self.prepared_data = ColumnTransformer([])

    def get_training_set(self):
        """
        :return: The training set
        """
        x_training_set, y_training_set = self.__split_into_x_y(self.training_set)
        self.feature_names, self.prepared_data = self.__fit_prep_pipeline(x_training_set)
        x_training_set = self.__transform(x_training_set, x_training_set.index)
        return x_training_set, y_training_set

    def get_test_set(self):
        """
        :return: The test set
        """
        x_test_set, y_test_set = self.__split_into_x_y(self.test_set)
        x_test_set = self.__transform(x_test_set, x_test_set.index)
        return x_test_set, y_test_set

    def __transform(self, data, index):
        data = self.prepared_data.transform(data)
        return pd.DataFrame(data, columns=self.feature_names, index=index)

    @staticmethod
    def __fit_prep_pipeline(data):
        cat_features = ["month", "weekday", "hour"]  # categorical features
        bool_features = ["holiday"]  # boolean features
        num_features = [c for c in data.columns
                        if c.startswith("load_lag")]  # numerical features
        prep_pipeline = ColumnTransformer([
            ("cat", OneHotEncoder(), cat_features),
            ("bool", FunctionTransformer(), bool_features),  # identity
            ("num", StandardScaler(), num_features),
        ])
        prep_pipeline = prep_pipeline.fit(data)

        feature_names = []
        one_hot_tf = prep_pipeline.transformers_[0][1]
        for i, cat_feature in enumerate(cat_features):
            categories = one_hot_tf.categories_[i]
            cat_names = [f"{cat_feature}_{c}" for c in categories]
            feature_names += cat_names
        feature_names += (bool_features + num_features)

        return feature_names, prep_pipeline

    @staticmethod
    def __split_into_x_y(data):
        target_col = "load"
        x = data.drop(columns=target_col)
        y = data.loc[:, target_col]
        logger.info("Data splitted into x and y")
        return x, y
