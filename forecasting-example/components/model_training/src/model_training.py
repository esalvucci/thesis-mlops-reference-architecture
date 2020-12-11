from xgboost import XGBRegressor
import pandas as pd
import fire


def train_model(x_training_set_path, y_training_set_path, x_test_set_path, y_test_set_path):
    n_estimators = 1000
    learning_rate = 0.5
    max_depth = 5

    x_training_set = pd.read_csv(x_training_set_path, index_col='time')
    y_training_set = pd.read_csv(y_training_set_path, index_col='time')
    x_test_set = pd.read_csv(x_test_set_path, index_col='time')
    y_test_set = pd.read_csv(y_test_set_path, index_col='time')

    model = XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_child_weight=5
    )

    model.fit(
        x_training_set, y_training_set, early_stopping_rounds=10,
        eval_set=[(x_training_set, y_training_set), (x_test_set, y_test_set)],
        verbose=False,
    )

#    pickle.dump(model, open('/tmp/trained_model.pkl', 'rb'))
    model.save_model('/tmp/trained_model.pkl')


if __name__ == "__main__":
    fire.Fire(train_model)
