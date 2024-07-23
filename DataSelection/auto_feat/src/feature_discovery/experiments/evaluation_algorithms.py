import logging
import time
from typing import Optional, List

import numpy as np

import pandas as pd
import typer as typer
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from sklearn.model_selection import train_test_split

from feature_discovery.config import AUTO_GLUON_FOLDER
from feature_discovery.experiments.dataset_object import REGRESSION
from feature_discovery.experiments.result_object import Result

hyper_parameters = [
    {"RF": {}},
    {"GBM": {}},
    {"XT": {}},
    {"XGB": {}},
    {'KNN': {}},
    {'LR': {'penalty': 'L1'}},
]


def get_hyperparameters(algorithm: Optional[str] = None) -> List[dict]:
    if algorithm is None:
        return hyper_parameters

    if algorithm == 'LR':
        return [{'LR': {'penalty': 'L1'}}]

    model = {algorithm: {}}
    if model in hyper_parameters:
        return [model]
    else:
        raise typer.BadParameter(
            "Unsupported algorithm. Choose one from the list: [RF, GBM, XT, XGB, KNN, LR]."
        )


def run_auto_gluon(dataframe: pd.DataFrame, target_column: str, problem_type: str, algorithms_to_run: dict, type: str = "autofeat"):
    from autogluon.tabular import TabularPredictor

    start = time.time()

    logging.debug(f"Train algorithms: {list(algorithms_to_run.keys())} with AutoGluon ...")

    selected_features = dataframe.columns

    if type != "base":
         if 'crypto_desktop\base_table.csv.user_id' in selected_features:
            users = dataframe['crypto_desktop\base_table.csv.user_id'].unique()
            total_num_users = len(users)
            X_train_users = np.random.choice(users, size=int(total_num_users*0.8))
            X_test_users = [user_id for user_id in users if (user_id not in X_train_users)]

            X_train = dataframe[dataframe["user_id"].isin(X_train_users)]
            X_test = dataframe[dataframe["user_id"].isin(X_test_users)]

            X_train = X_train.drop("user_id", axis=1)
            X_test = X_test.drop("user_id", axis=1)
         else:
            X_train, X_test = train_test_split(
                dataframe,
                test_size=0.2,
                random_state=10,
            )
                    
            
    # this will do train test split with X and y, by removing X from y



    # X_train, X_test, y_train, y_test = train_test_split(
    #     dataframe.drop(columns=[target_column]),
    #     dataframe[[target_column]],
    #     test_size=0.2,
    #     random_state=10,
    # )

    # X_train, X_test = train_test_split(
    #     dataframe,
    #     test_size=0.2,
    #     random_state=10,
    # )

    # train test split with respect to users

# if running the baseline, we have this problem
    else:
        users = dataframe["user_id"].unique()
        total_num_users = len(users)
        X_train_users = np.random.choice(users, size=int(total_num_users*0.8))
        X_test_users = [user_id for user_id in users if (user_id not in X_train_users)]

        X_train = dataframe[dataframe["user_id"].isin(X_train_users)]
        X_test = dataframe[dataframe["user_id"].isin(X_test_users)]

        X_train = X_train.drop(["user_id", "base_key"], axis=1)
        X_test = X_test.drop(["user_id", "base_key"], axis=1)



    # print(X_train.head())

    # y_train =  X_train["label"]
    # y_test = X_test["label"]
    # print(X_train.head())
    # print(y_train.head())

    #         # split the training data with respect to user_id

    # X_train = dataframe.loc[dataframe["user_id"] ]

    # this will join the train test split back into the dataframe
    join_path_features = list(X_train.columns)
    # X_train[target_column] = y_train
    # X_test[target_column] = y_test

    predictor = TabularPredictor(label=target_column,
                                 problem_type=problem_type,
                                 verbosity=0,
                                 path=AUTO_GLUON_FOLDER / "models").fit(train_data=X_train,
                                                                        hyperparameters=algorithms_to_run)
    
    score_types = ['accuracy', "precision", "recall"]
    if problem_type == REGRESSION:
        score_type = ['root_mean_squared_error']

    results = []

    model_names = predictor.get_model_names()
    for model in model_names[:-1]:
        # for score_type in score_types:

        result = predictor.evaluate(data=X_test, model=model)

        metric_dict = {}
        for score_type in score_types:
            metric_dict[score_type] = result[score_type]

        ft_imp = predictor.feature_importance(
            data=X_test, model=model, feature_stage="original"
        )

        entry = Result(
            algorithm=model,
            **metric_dict,
            feature_importance=dict(zip(list(ft_imp.index), ft_imp["importance"])),
            join_path_features=join_path_features,
        )

        results.append(entry)

        break


    end = time.time()


    return end - start, results


def evaluate_all_algorithms(dataframe: pd.DataFrame, target_column: str, algorithm: str, problem_type: str = 'binary', run_type :str = "autofeat"):
    
    # when running the baseline, then need to remove the foreign keys
    # print(dataframe.head())
    # dataframe = dataframe.drop(['base_key', 'user_id'], axis=1)
    # print(dataframe.head())

    hyperparams = get_hyperparameters(algorithm)
    all_results = []
    df = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False
    ).fit_transform(X=dataframe)

    logging.debug(f"Training AutoGluon ... ")
    for model in hyperparams:
        runtime, results = run_auto_gluon(
            dataframe=df,
            target_column=target_column,
            algorithms_to_run=model,
            problem_type=problem_type,
            type = run_type
        )

        for res in results:
            res.train_time = runtime
            res.total_time += res.train_time
        all_results.extend(results)

    return all_results, df
