import use_cases.columns as col
import src.utils.utils as ut
import models.model as md


if __name__ == "__main__":
    smote = True

    # For IoT dataset
    path_train = "/home/santiubuntu/Escritorio/projects/extreme_xp_smote/dataset/IoT/train.dataset"
    path_test = "/home/santiubuntu/Escritorio/projects/extreme_xp_smote/dataset/IoT/test.dataset"
    data_train = ut.load_pickle(path_train)
    data_test = ut.load_pickle(path_test)
    
    data_train["label"] = data_train["label"].map({"Bening": 0, "Malicious": 1})
    data_test["label"] = data_test["label"].map({"Bening": 0, "Malicious": 1})

    md.train_model(data_train, data_test, smote, col.COLUMNS_IOT, debug=True)
