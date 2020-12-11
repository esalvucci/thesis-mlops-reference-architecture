import pickle

from google.cloud import storage
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import fire


def evaluate_model(dataset_name, model_path):
    bucket_name = 'kubeflow-demo'
    folder_path = 'forecast-example'
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob('it.csv')

    model = pickle.load(open(model_path, 'wb'))

    x_dataset = pd.read_csv(x_dataset_name, index_col='time')
    y_dataset = pd.read_csv(y_dataset_path, index_col='time')

    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(model, x_dataset, y_dataset, cv=kfold)
    accuracy_as_string = "Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)

    with open('/tmp/accuracy.txt', 'w') as output_text:
        output_text.write(accuracy_as_string)


if __name__ == "__main__":
    fire.Fire(evaluate_model)
