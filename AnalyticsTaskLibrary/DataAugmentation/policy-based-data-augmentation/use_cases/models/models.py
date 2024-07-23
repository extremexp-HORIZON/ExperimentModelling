import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import src.data_augmentation.smote_algorithm as smote_alg
import src.data_augmentation.ctgan_algorithm as ctgan_alg
import src.utils.metrics_model as mtr_model
import src.utils.utils as ut


def create_model() -> GridSearchCV:

    fix_params = {'n_estimators': 100, 'learning_rate': 0.1, 'objective': 'binary:logistic', 'random_state': ut.SEED}
    cv_params = {'max_depth': [1, 2, 3, 4, 5, 6], 'min_child_weight': [1, 2, 3, 4]}   
    csv = GridSearchCV(xgb.XGBClassifier(**fix_params), cv_params, scoring="f1", n_jobs=5)
    return csv


def train_model(data_train: pd.DataFrame, data_test: pd.DataFrame, augmentation_type: str, feat_columns: list, debug: bool = False):
 
    if debug:
        #print(f"[DEBUG] Columns used {feat_columns}")
        print(f'[DEBUG] Labels before data augemntation: \n {data_train["label"].value_counts()}')

    data_train = apply_data_augmentation(data_train, feat_columns, augmentation_type)

    x_train = data_train[feat_columns]
    y_train = data_train["label"]

    if debug:
        print(f'[DEBUG] Labels after data augmentation: {y_train.value_counts()}')

    x_test = data_test[feat_columns]
    y_test = data_test["label"]

    csv = create_model()
    csv.fit(x_train, y_train)

    model = csv.best_estimator_
    y_pred = model.predict(x_test)

    metrics = mtr_model.calculate_metrics(y_test, y_pred)

    print(metrics)


if __name__ == "__main__":

    augmentation_type = "ctgan" # None for no augemntation smote, ctgan, None

    # For phishing dataset
    path_train = "/home/santiubuntu/Escritorio/projects/extreme_xp_smote/dataset/phishing/feat_vectors_train.pickle"
    path_test = "/home/santiubuntu/Escritorio/projects/extreme_xp_smote/dataset/phishing/feat_vectors_test.pickle"
    data_train = ut.load_pickle(path_train)
    data_test = ut.load_pickle(path_test)
    train_model(data_train, data_test, augmentation_type, ut.COLUMNS_PHISHING, debug=True)


 