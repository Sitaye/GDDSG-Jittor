import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import numpy as np

def predict_ideal_taskid(train_data, test_data):
    X_train = train_data.drop(['label', 'orginal_label'], axis=1)
    y_train = train_data['label'] 
    X_test = test_data.drop(['label', 'orginal_label'], axis=1)
    y_test = test_data['label']
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
    smote = SMOTE(random_state=42)
    try:
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
        with tqdm(total=1, desc="Training and Evaluation") as pbar:
            pbar.set_description("Training and Evaluation")
            top_accuracy_scores = []
            for _ in range(1):
                clf = RandomForestClassifier()
                clf.fit(X_train_resampled, y_train_resampled)
                y_pred = clf.predict(X_test)
                top_accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
                top_accuracy_scores.append(top_accuracy)
                pbar.update(1)
    except ValueError:
        with tqdm(total=1, desc="Training and Evaluation") as pbar:
            pbar.set_description("Training and Evaluation")
            top_accuracy_scores = []
            for _ in range(1):
                clf = RandomForestClassifier()
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                top_accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
                top_accuracy_scores.append(top_accuracy)
                pbar.update(1)
    return y_pred

