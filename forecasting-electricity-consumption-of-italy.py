#!/usr/bin/env python
# coding: utf-8

# # MLFlow Demo - Forecasting hourly electricity consumption of Italy

# [From Kaggle - "Forecasting electricity consumption of Germany" Notebook](https://www.kaggle.com/francoisraucent/forecasting-electricity-consumption-of-germany)
# 
# Electricity grid and market have become increasingly challenging to operate and maintain in the recent years. In particular, one of the main responsibilities of transmission system operators and aggregators consists in maintaining balance between production and consumption. With the development and spread of renewable energy sources, production has become more intermittent which requires even more effort to maintain the balance.
# 
# One of the underlying task of maintaining grid balance, it to forecast the consumption. In this analysis, we train and test a few models to forecast total german load, on an hourly basis, with a lead time of 24 hours.
# 
# The data was retrieved from [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), which provides access to electricity generation, transportation, and consumption data for the pan-European market.
# 
# This notebook follows a structure similar to [this nice tutorial](https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-xgboost) from [Rob Mulla](https://www.kaggle.com/robikscube). The main adaptations are the following :
# * it applies to German load instead of PJM data covering US east region,
# * it includes additional features such as holidays and lag features,
# * a linear model, and a random forest are used as baselines in addition to the XGB model,
# * the final XGB model is finetuned with some grid search CV.
import pandas as pd
import numpy as np
import holidays
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split, KFold, GridSearchCV, ParameterGrid,
)
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, DMatrix, plot_importance
from xgboost import cv as xgb_cv
from urllib.parse import urlparse
import sys
import mlflow
import os

# ## Loading the data
# We will work with consumption data ranging from Jan 2015 to Jan 2020.
STUDY_START_DATE = pd.Timestamp("2015-01-01 00:00", tz="utc")
STUDY_END_DATE = pd.Timestamp("2020-01-31 23:00", tz="utc")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../../mlflow-demo-294110-b2aae8662969.json'

# The German load data is originally available with 15-min resolution. We have resampled it on an hourly basis for this analysis.


def split_train_test(df, split_time):
    df_train = df.loc[df.index < split_time]
    df_test = df.loc[df.index >= split_time]
    return df_train, df_test


def compute_learning_curves(model, X, y, curve_step, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    n_train_obs = X_train.shape[0]

    ###                                                 ###
    #    n_iter = math.ceil(n_train_obs / curve_step)     #
    ###                                                 ###
    n_iter = 1
    train_errors, val_errors, steps = [], [], []
    for i in range(n_iter):
        n_obs = (i + 1) * curve_step
        n_obs = min(n_obs, n_train_obs)
        model.fit(X_train[:n_obs], y_train[:n_obs])

        y_train_predict = model.predict(X_train[:n_obs])
        y_val_predict = model.predict(X_val)

        train_mse = mean_squared_error(y_train[:n_obs], y_train_predict)
        val_mse = mean_squared_error(y_val, y_val_predict)

        train_errors.append(train_mse)
        val_errors.append(val_mse)
        steps.append(n_obs)
        if verbose:
            msg = "Iteration {0}/{1}: train_rmse={2:.2f}, val_rmse={3:.2f}".format(
                i + 1, n_iter, np.sqrt(train_mse), np.sqrt(val_mse)
            )
            print(msg)
    return steps, train_errors, val_errors


def plot_predictions(pred_df, start=None, end=None):
    figure, ax = plt.subplots(1, 1, figsize=(12, 5))

    start = start or pred_df.index.min()
    end = end or pred_df.index.max()
    pred_df.loc[
        (pred_df.index >= start) & (pred_df.index <= end),
        ["actual", "prediction"]
    ].plot.line(ax=ax)
    ax.set_title("Predictions on test set")
    ax.set_ylabel("MW")
    ax.grid()

def compute_predictions_df(model, X, y):
    y_pred = model.predict(X)
    df = pd.DataFrame(dict(actual=y, prediction=y_pred), index=X.index)
    df["squared_error"] =  (df["actual"] - df["prediction"])**2
    return df


def add_time_features(df):
    cet_index = df.index.tz_convert("CET")
    df["month"] = cet_index.month
    df["weekday"] = cet_index.weekday
    df["hour"] = cet_index.hour
    return df


def add_holiday_features(df):
    it_holidays = holidays.Italy()
    cet_dates = pd.Series(df.index.tz_convert("CET"), index=df.index)
    df["holiday"] = cet_dates.apply(lambda d: d in it_holidays)
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


def plot_learning_curves(steps, train_errors, val_errors, ax=None, title=""):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))
    train_rmse = np.sqrt(train_errors)
    val_rmse = np.sqrt(val_errors)
    ax.plot(steps, train_rmse, color="tab:blue",
            marker=".", label="training")
    ax.plot(steps, val_rmse, color="tab:orange",
            marker=".", label="validation")
    ylim = (0.8*np.median(train_rmse),
            1.5*np.median(val_rmse))
    ax.set_ylim(ylim)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("RMSE (MW)")
    ax.set_title(title)
    ax.legend()
    ax.grid()


def xgb_grid_search_cv(
    params_grid, X, y, nfold,
    num_boost_round=1000, early_stopping_rounds=10,
):
    params_grid = ParameterGrid(params_grid)
    search_results = []
    print(f"Grid search CV : nfold={nfold}, " +
          f"numb_boost_round={num_boost_round}, " +
          f"early_stopping_round={early_stopping_rounds}")
    for params in params_grid:
        print(f"\t{params}")
        cv_df = xgb_cv(
            params=params, dtrain=DMatrix(X, y), nfold=nfold,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            shuffle=False, metrics={"rmse"},
        )
        cv_results = params.copy()
        cv_results["train-rmse-mean"] = cv_df["train-rmse-mean"].min()
        cv_results["test-rmse-mean"] = cv_df["test-rmse-mean"].min()
        search_results.append(cv_results)
    return pd.DataFrame(search_results)


it_load = pd.read_csv("components/data_preparation/datasets/it.csv")
it_load = it_load.drop(columns="end").set_index("start")
it_load.index = pd.to_datetime(it_load.index)
it_load.index.name = "time"
it_load = it_load.groupby(pd.Grouper(freq="h")).mean()
it_load = it_load.loc[
    (it_load.index >= STUDY_START_DATE) & (it_load.index <= STUDY_END_DATE), :
]
it_load.info()

df_train, df_test = split_train_test(
    it_load, pd.Timestamp("2019-02-01", tz="utc")
)

ax = df_train["load"].plot(figsize=(12, 4), color="tab:blue")
_ = df_test["load"].plot(ax=ax, color="tab:orange", ylabel="MW")


# ## Data preparation

# There are no missing observations in our training data (there actually were a few missing observations on 15-min granularity, but we took care of these with hourly aggregation when loading the data).

df_train.loc[df_train["load"].isna(), :].index

# ### Create features for training

# The following features are used for training our forecast models :
# * time features: month, weekday and hour
# * national holiday features, as a boolean time series
# * lag features: load data with a lag values ranging from 24 to 48 hours

# In[27]:

# The lag features introduce a few missing values which we will move out of the analysis. The features of our training set are then the following :

# In[28]:


df_train = add_all_features(df_train).dropna()
df_test = add_all_features(df_test).dropna()
df_train.info()


# We then separate target values from features into distinct data frames.

# In[29]:


target_col = "load"
X_train = df_train.drop(columns=target_col)
y_train = df_train.loc[:, target_col]
X_test = df_test.drop(columns=target_col)
y_test = df_test.loc[:, target_col]


# ### Data preparation pipeline

# We'll use the following data preparation pipeline to apply one-hot encoders on categorical feratures (time features), and a standard scaler on numerical features (lag features).

# In[30]:

# We then fit pipeline on training data, and apply it on training and test sets

# In[31]:


feature_names, prep_pipeline = fit_prep_pipeline(X_train)

X_train_prep = prep_pipeline.transform(X_train)
X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names, index=df_train.index)
X_test_prep = prep_pipeline.transform(X_test)
X_test_prep = pd.DataFrame(X_test_prep, columns=feature_names, index=df_test.index)

X_train_prep.info()

xgb_model = XGBRegressor(n_estimators=1000)

# XGBoost model pushes RMSE even further to 2050MW with training RMSE of only 260MW. This indicates once again that our model is overfitting the training data. We will try to handle the overfitting later on at model fine-tuning step.

xgb_steps, xgb_train_mse, xgb_val_mse = compute_learning_curves(
    xgb_model, X_train_prep, y_train, 500, verbose=True
)

plot_learning_curves(xgb_steps, xgb_train_mse, xgb_val_mse, title="XGB")

params_grid = dict(
    eta = [0.05, 0.1, 0.3],
    max_depth = [2, 4, 6],
    min_child_weight = [5, 1]
)

# xgb_search_scores = xgb_grid_search_cv(
#    params_grid, X_train_prep, y_train, nfold=4, early_stopping_rounds=10
# )

# Results of the grid search are the following :

# xgb_search_scores.sort_values(by="test-rmse-mean")

n_estimators = 1000
learning_rate = 0.5
max_depth = 6
min_child_weight = 5

experiment_name = "MLFlow Demo"
## check if the experiment already exists
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

#mlflow.set_tracking_uri(tracking_uri)

# Three models will be trained for our prediction task : a simple linear models with L1 and L2 regularisation, a random forest, and gradient boosting model (based on XGBoost library).
with mlflow.start_run(experiment_id=experiment.experiment_id):
    #    mlflow.set_experiment('/mlflow_test/electricity-consumption-of-italy')

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_child_weight", min_child_weight)

    # ### Training our final model
    # Based on previous result, we can train our final model on the whole training set

    final_model = XGBRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=5
    )
    final_model.fit(
        X_train_prep, y_train, early_stopping_rounds=10,
        eval_set=[(X_train_prep, y_train), (X_test_prep, y_test)],
        verbose=False,
    )

    # ## Predictions on test set

    # Our final XGB model achieves RMSE score of ±1740MW on test set
    mlflow.log_metric("rmse", final_model.best_score)

    import mlflow.xgboost

    # Register the model
    mlflow.xgboost.log_model(final_model, "XGB_regressor_model")

    # Let's group predicted and actual test data into a data frame
    pred_df = compute_predictions_df(
        final_model, X_test_prep, y_test
    )
    pred_df.head()

    # Comparing actual and predicted curves on the test set :
#    plot_predictions(pred_df)

    # ### Special time intervals

    # The intervals
    # * 15th Apr. 2019 – 6th May 2019, and
    # * 16th Dec. 2019 – 6th Jan 2020
    #
    # seem slightly irregular compared to the rest. This is most likely due to holiday periods. Let's zoom on these periods  to check how our predictions perform.

    plot_predictions(pred_df,
                     start=pd.Timestamp("2019-04-15", tz="utc"),
                     end=pd.Timestamp("2019-05-06", tz="utc"))


    # Performance drops slightly around 21st of April and on May 2nd. For the 2nd of May, this could be explained by the fact that 1st of May is a national holiday in Germany, and that the model is using lag features to estimate May 2nd volume using May 1st without properly adapting scale (as May 2nd is not a holiday).
    #
    # Appart from these, our model does a reasonable job at predicting volumes.
    plot_predictions(pred_df,
                     start=pd.Timestamp("2019-12-16", tz="utc"),
                     end=pd.Timestamp("2020-01-06", tz="utc"))


    # Here we observe that our model is slightly overestimating consumption during Christmas period and New Year's Eve.

    # ### Best and worst prediction days

    # The days with worst prediction performance are the following :
    daily_pred_df = pred_df.groupby(pd.Grouper(freq="D")).mean()
    daily_pred_df.sort_values(by="squared_error", ascending=False).head(5)

    # Our model is largely overestimating consumption on 20th of June. This date is actually a *regional* holiday in Germany (Fronleichnam) which are currently not taken into account by our model. These only and happen in specific states, but still have an impact on the load at country level.
    #
    # Looking at the best predictions, we have the following days :
    daily_pred_df.sort_values(by="squared_error", ascending=True).head(5)

    # Predictions were particularly accurate on 23rd and 25th of January 2020, as we can see from the graph below.
    plot_predictions(pred_df,
                     start=pd.Timestamp("2020-01-23", tz="utc"),
                     end=pd.Timestamp("2020-01-26", tz="utc"))
