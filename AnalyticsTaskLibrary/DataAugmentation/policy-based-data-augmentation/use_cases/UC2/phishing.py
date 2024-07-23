import use_cases.columns as col
import src.utils.utils as ut
import models.model as md
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(dataset: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    # Get unique user IDs
    unique_users = dataset['user_id'].unique()

    # Split user IDs into train and test sets
    users_train, users_test = train_test_split(unique_users, test_size=0.3,
                                               random_state=ut.SEED)

    # Filter data for train and test sets based on user IDs
    data_train = dataset[dataset['user_id'].isin(users_train)]
    data_test = dataset[dataset['user_id'].isin(users_test)]

    return data_train, data_test


if __name__ == "__main__":
    data_augmentation = "ctgan"
 
    # For phishing dataset
    path_train = "/home/santiubuntu/Escritorio/projects/behavior-datasets/phishing_behaviour/min_windows_size_6/smartphone_features.pckl"
 
    dataset = ut.load_pickle(path_train)
    data_train, data_test = split_dataset(
                            dataset.loc[~(dataset["label"] == -1), :])
    columns_to_exclude = ["user_id", "entity", "label", "timestamp"]
    columns = dataset.columns.drop(columns_to_exclude).to_list()

    md.train_model(data_train, data_test, data_augmentation, columns, debug=True)
