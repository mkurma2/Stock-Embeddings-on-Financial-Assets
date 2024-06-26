import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


def get_sector_score(
    embedding_matrix: np.ndarray,
    sectors: list,
    classifier: SVC = SVC(kernel="rbf", probability=True),
    smote: bool = True,
    top_k_accuracy: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
    k: int = 3,
) -> None:
    """
    Calculate various scores for the sector classification.

    :param embedding_matrix: Input data for training and testing.
    :param sectors: The target sectors.
    :param classifier: A classifier object, defaults to SVC.
    :param smote: Boolean flag to apply SMOTE, defaults to True.
    :param top_k_accuracy: Boolean flag to calculate top-k accuracy.
    :param n_splits: Number of splits for cross-validation.
    :param random_state: Random state for reproducibility.
    :param k: The 'k' in top-k accuracy.
    """

    accuracy_list = []
    accuracy_list_top_k = []
    f1_list, recall_list, precision_list = [], [], []

    X = embedding_matrix
    y = np.array(sectors).reshape(-1, 1)

    unique_strings, counts = np.unique(y, return_counts=True)
    indices_to_keep = np.where(y != "UNKNOWN")[0]

    X = X[indices_to_keep]
    y = y[indices_to_keep]


    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_index, test_index in kf.split(X, y.flatten()):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        if smote:
            sm = SMOTE(k_neighbors=1)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        classifier.fit(X_train, y_train.ravel())
        y_pred = classifier.predict(X_test)

        accuracy_list.append(classifier.score(X_test, y_test))
        if top_k_accuracy:
            accuracy_list_top_k.append(
                top_k_accuracy_score(y_test, classifier.predict_proba(X_test), k=k)
            )
        f1_list.append(f1_score(y_test, y_pred, average="weighted"))
        recall_list.append(recall_score(y_test, y_pred, average="weighted"))
        precision_list.append(precision_score(y_test, y_pred, average="weighted"))

    print(f"Precision Score: {np.round(np.mean(precision_list), 2)}")
    print(f"Recall Score: {np.round(np.mean(recall_list), 2)}")
    print(f"F1 Score: {np.round(np.mean(f1_list), 2)}")
    print(f"Accuracy Score: {np.round(np.mean(accuracy_list), 2)}")
    if top_k_accuracy:
        print(f"Accuracy Score Top-{k}: {np.round(np.mean(accuracy_list_top_k), 2)}")
