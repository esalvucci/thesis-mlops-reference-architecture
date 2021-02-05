import numpy as np
import holidays
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
#        self.training_set, self.test_set = train_test_split(self.df, test_size=0.33)
        self.feature_names = []
        self.column_transformer = ColumnTransformer([])
        self.df = self.__add_all_features(self.df).dropna()
        self.feature_names, self.column_transformer = self.__fit_column_transformer(self.df)

    def get_transformed_dataset(self):
        """
        :return:
        """
        data = self.column_transformer.transform(self.df)
        columns = self.feature_names
        return pd.DataFrame(data, columns=columns, index=self.df.index)

    def __add_all_features(self, df, target_column="load"):
        df = df.copy()
        df = self.__add_time_features(df)
        df = self.__add_holiday_features(df)
        if 'load' in df.columns:
            df = self.__add_lag_features(df, column=target_column)
        return df

    @staticmethod
    def __fit_column_transformer(data):
        """
        Apply one-hot encoders on categorical features (time features), and a standard scaler on numerical features
        :param data - The data to apply the encoders and the scaler. Input data, of which specified subsets
                      are used to fit the transformers
        :return: The feature names and the column transformer
        """
        categorical_features = ["month", "weekday", "hour"]
        boolean_features = ["holiday"]
        numerical_features = [c for c in data.columns if c.startswith("load_lag")]
        column_transformer = ColumnTransformer([
            ("cat", OneHotEncoder(), categorical_features),
            ("bool", FunctionTransformer(), boolean_features),
            ("num", StandardScaler(), numerical_features),
        ])
        column_transformer = column_transformer.fit(data)
        feature_names = []
        one_hot_tf = column_transformer.transformers_[0][1]
        for i, cat_feature in enumerate(categorical_features):
            categories = one_hot_tf.categories_[i]
            cat_names = [f"{cat_feature}_{c}" for c in categories]
            feature_names += cat_names
        feature_names += (boolean_features + numerical_features)
        return feature_names, column_transformer

    @staticmethod
    def __add_time_features(df):
        cet_index = df.index.tz_convert("CET")
        df["month"] = cet_index.month
        df["weekday"] = cet_index.weekday
        df["hour"] = cet_index.hour
        return df

    @staticmethod
    def __add_holiday_features(df):
        df.index = pd.to_datetime(df.index)
        df_holidays = holidays.Germany()
        cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
        df["holiday"] = cet_dates.apply(lambda d: d in df_holidays)
        df["holiday"] = df["holiday"].astype(int)
        return df

    @staticmethod
    def __add_lag_features(df, column="load"):
        for n_hours in range(24, 49):
            shifted_col = df[column].shift(n_hours, "h")
            shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
            label = f"{column}_lag_{n_hours}"
            df[label] = np.nan
            df.loc[shifted_col.index, label] = shifted_col
        return df
