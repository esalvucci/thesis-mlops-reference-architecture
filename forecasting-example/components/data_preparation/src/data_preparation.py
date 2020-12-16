import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, FunctionTransformer)
from sklearn.compose import ColumnTransformer
import os
from singleton_logger import SingletonLogger

logger = SingletonLogger.get_logger()


def split_train_test(df, split_time):
    train_set = df.loc[df.index < split_time]
    test_set = df.loc[df.index > split_time]
    return train_set, test_set


def add_time_features(df):
    cet_index = df.index.tz_convert("CET")
    df["month"] = cet_index.month
    df["weekday"] = cet_index.weekday
    df["hour"] = cet_index.hour
    return df


def add_holiday_features(df):
    de_holidays = holidays.Germany()
    cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
    df["holiday"] = cet_dates.apply(lambda d: d in de_holidays)
    df["holiday"] = df["holiday"].astype(int)
    return df


def add_lag_features(df, col="load"):
    for n_hours in range(24, 49):
        shifted_col = df[col].shift(n_hours, "h")
        shifted_col = shifted_col.loc[df.index.min(): df.index.max()]
        label = f"{col}_lag_{n_hours}"
        df[label] = np.nan
        df.loc[shifted_col.index, label] = shifted_col
    return df


def add_all_features(df, target_col="load"):
    df = df.copy()
    df = add_time_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df, col=target_col)
    return df


def fit_prep_pipeline(df):
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


STUDY_START_DATE = pd.Timestamp("2015-01-01 00:00", tz="utc")
STUDY_END_DATE = pd.Timestamp("2020-01-31 23:00", tz="utc")
dataset_url = os.path.join(os.getcwd(), "datasets/it.csv")
it_load = pd.read_csv(dataset_url)
it_load = it_load.drop(columns="end").set_index("start")
it_load.index = pd.to_datetime(it_load.index, errors='coerce')
it_load.index.name = "time"
it_load = it_load.groupby(pd.Grouper(freq="h")).mean()
it_load = it_load.loc[
    (it_load.index >= STUDY_START_DATE) & (it_load.index <= STUDY_END_DATE), :
]

df_train, df_test = split_train_test(
    it_load, pd.Timestamp("2019-02-01", tz="utc")
)

# Artifacts
ax = df_train["load"].plot(figsize=(12, 4), color="tab:blue")
plt.savefig('/tmp/training_set.png')

_ = df_test["load"].plot(ax=ax, color="tab:orange", ylabel="MW")
plt.savefig('/tmp/test_set.png')

var = df_train.loc[df_train["load"].isna(), :].index
df_train = add_all_features(df_train).dropna()
df_test = add_all_features(df_test).dropna()

target_col = "load"
X_train = df_train.drop(columns=target_col)
y_train = df_train.loc[:, target_col]

X_test = df_test.drop(columns=target_col)
y_test = df_test.loc[:, target_col]

feature_names, prep_pipeline = fit_prep_pipeline(X_train)

X_train_prep = prep_pipeline.transform(X_train)
X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names, index=df_train.index)

X_test_prep = prep_pipeline.transform(X_test)
X_test_prep = pd.DataFrame(X_test_prep, columns=feature_names, index=df_test.index)

X_train_prep.to_csv('/tmp/x_train.csv')
logger.info("X Training set saved")

y_train.to_csv('/tmp/y_train.csv')
logger.info("Y Training set saved")

X_test_prep.to_csv('/tmp/x_test.csv')
logger.info("X Test set saved")

y_test.to_csv('/tmp/y_test.csv')
logger.info("Y Test set saved")

# Upload the prepared data to gcp bucket
#bucket_name = 'kubeflow-demo'
#folder_path = 'forecast-example'
#client = storage.Client()
#bucket = client.get_bucket(bucket_name)

#bucket.blob(os.path.join(folder_path, 'x_train.csv')).upload_from_string(X_train_prep.to_csv(), 'text/csv')
#bucket.blob(os.path.join(folder_path, 'y_train.csv')).upload_from_string(y_train.to_csv(), 'text/csv')
#bucket.blob(os.path.join(folder_path, 'x_test.csv')).upload_from_string(X_test_prep.to_csv(), 'text/csv')
#bucket.blob(os.path.join(folder_path, 'y_test.csv')).upload_from_string(y_test.to_csv(), 'text/csv')
