import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, FunctionTransformer)
from sklearn.compose import ColumnTransformer
from singleton_logger import SingletonLogger
import fire

logger = SingletonLogger.get_logger()
STUDY_START_DATE = pd.Timestamp("2015-01-01 00:00", tz="utc")
STUDY_END_DATE = pd.Timestamp("2020-01-31 23:00", tz="utc")


def __split_train_test(df, split_time):
    train_set = df.loc[df.index < split_time]
    test_set = df.loc[df.index > split_time]
    return train_set, test_set


def __add_time_features(df):
    cet_index = df.index.tz_convert("CET")
    df["month"] = cet_index.month
    df["weekday"] = cet_index.weekday
    df["hour"] = cet_index.hour
    return df


def __add_holiday_features(df):
    de_holidays = holidays.Germany()
    cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
    df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
    df["holiday"] = df["holiday"].astype(int)
    return df


def __add_lag_features(df, col="load"):
    for n_hours in range(24, 49):
        shifted_col = df[col].shift(n_hours, "h")
        shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
        label = f"{col}_lag_{n_hours}"
        df[label] = np.nan
        df.loc[shifted_col.index, label] = shifted_col
    return df


def __add_all_features(df, target_col="load"):
    df = df.copy()
    df = __add_time_features(df)
    df = __add_holiday_features(df)
    df = __add_lag_features(df, col=target_col)
    return df


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


def __save_train_test_set_artifacts(train_set, test_set):
    ax = train_set["load"].plot(figsize=(12, 4), color="tab:blue")
    plt.savefig('/tmp/training_set.png')

    _ = test_set["load"].plot(ax=ax, color="tab:orange", ylabel="MW")
    plt.savefig('/tmp/test_set.png')


def __split_data_into_x_y(data):
    target_col = "load"
    x = data.drop(columns=target_col)
    y = data.loc[:, target_col]
    return x, y


def __prepare_data(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.drop(columns="end").set_index("start")
    df.index = pd.to_datetime(df.index, errors='coerce')
    df.index.name = "time"
    df = df.groupby(pd.Grouper(freq="h")).mean()
    df = df.loc[
         (df.index >= STUDY_START_DATE) & (df.index <= STUDY_END_DATE), :
         ]

    df_train, df_test = __split_train_test(
        df, pd.Timestamp("2019-02-01", tz="utc")
    )

    __save_train_test_set_artifacts(df_train, df_test)

    var = df_train.loc[df_train["load"].isna(), :].index
    df_train = __add_all_features(df_train).dropna()
    df_test = __add_all_features(df_test).dropna()

    x_train, y_train = __split_data_into_x_y(df_train)
    x_test, y_test = __split_data_into_x_y(df_test)

    feature_names, prep_pipeline = __fit_prep_pipeline(x_train)

    x_train_prep = prep_pipeline.transform(x_train)
    x_train_prep = pd.DataFrame(x_train_prep, columns=feature_names, index=df_train.index)

    x_test_prep = prep_pipeline.transform(x_test)
    x_test_prep = pd.DataFrame(x_test_prep, columns=feature_names, index=df_test.index)

    x_train_prep.to_csv('/tmp/x_train.csv')
    logger.info("X Training set saved")

    y_train.to_csv('/tmp/y_train.csv')
    logger.info("Y Training set saved")

    x_test_prep.to_csv('/tmp/x_test.csv')
    logger.info("X Test set saved")

    y_test.to_csv('/tmp/y_test.csv')
    logger.info("Y Test set saved")


if __name__ == "__main__":
    fire.Fire(__prepare_data)
