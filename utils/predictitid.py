import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
import optuna

optuna.logging.enable_propagation
warnings.filterwarnings('ignore')

def train_model(train_data):
    """
    Input:
        train_data (pandas.DataFrame): A DataFrame containing the training dataset. This dataset must include
            feature columns as well as 'label' and 'orginal_label' columns, where 'label' represents the target
            variable used for model training.
    Output:
        tuple: A tuple containing:
            - models (list): A list of three trained classifier models [RandomForestClassifier, KNeighborsClassifier, LGBMClassifier].
            - best_a (float): The optimal ensemble weight for the RandomForestClassifier.
            - best_b (float): The optimal ensemble weight for the KNeighborsClassifier.
            - best_c (float): The optimal ensemble weight for the LGBMClassifier, computed as 1 - best_a - best_b.
    Function:
        This function orchestrates the training of an ensemble of three classifiers (RandomForest, KNeighbors,
        and LGBM) on the provided training dataset. It first splits the dataset into training and validation folds
        using stratified sampling. Subsequently, each classifier is trained on the training fold. A nested objective
        function is defined to optimize the ensemble weights using Optuna, aiming to maximize the training accuracy
        by combining the prediction probabilities of the three classifiers. The optimal weights and trained models
        are then returned.
    """
    X_train = train_data.drop(['label', 'orginal_label'], axis=1)
    y_train = train_data['label']
    
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    smote = SMOTE(random_state=42)
    
    rf_model = RandomForestClassifier(
        bootstrap=False,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )
    rf_model.fit(X_train_fold, y_train_fold)

    knn_model = KNeighborsClassifier(
        metric='euclidean',
        n_neighbors=3,
        weights='distance'
    )
    knn_model.fit(X_train_fold, y_train_fold)
    
    lgb_model = LGBMClassifier(
        colsample_bytree=0.8,
        learning_rate=0.1,
        max_depth=-1,
        n_estimators=200,
        num_leaves=100,
        subsample=0.8,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train_fold, y_train_fold)
    
    def objective(trial):
        """
        Input:
            trial (optuna.trial.Trial): An Optuna trial object used for suggesting hyperparameters during optimization.
        Output:
            float: The accuracy score on the entire training dataset obtained by combining the prediction probabilities
                of the three classifiers using the suggested ensemble weights.
        Function:
            This nested function defines the objective for hyperparameter optimization with Optuna. It samples two
            ensemble weight parameters, 'a' and 'b', while the third weight 'c' is calculated as 1 - a - b. Using these
            weights, it computes a weighted combination of the prediction probabilities from the pre-trained models on the
            training set. The class predictions are then obtained by selecting the class with the highest weighted probability,
            and the resulting accuracy score is returned as the objective to maximize.
        """
        a = trial.suggest_float('a', 0.0, 1.0)
        b = trial.suggest_float('b', 0.0, 1.0)
        c = 1 - a - b 
        
        p1 = rf_model.predict_proba(X_train)
        p2 = knn_model.predict_proba(X_train)
        p3 = lgb_model.predict_proba(X_train)
        
        weighted_p = a * p1 + b * p2 + c * p3
        y_pred = np.argmax(weighted_p, axis=1)
        
        accuracy = accuracy_score(y_train, y_pred)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_a = study.best_params['a']
    best_b = study.best_params['b']
    best_c = 1 - best_a - best_b 
    print(f"Best parameters: a={best_a}, b={best_b}, c={best_c}")
    print(f"Best accuracy: {study.best_value}")
    
    return [rf_model, knn_model, lgb_model], best_a, best_b, best_c

def test_model(models, test_data, best_a, best_b, best_c):
    """
    Input:
        models (list): A list containing three trained classifier models [RandomForestClassifier, KNeighborsClassifier, LGBMClassifier].
        test_data (pandas.DataFrame): A DataFrame containing the test dataset. This dataset must include feature columns
            as well as 'label' and 'orginal_label' columns, where 'label' represents the ground truth for evaluation.
        best_a (float): The optimal ensemble weight for the RandomForestClassifier as determined during training.
        best_b (float): The optimal ensemble weight for the KNeighborsClassifier as determined during training.
        best_c (float): The optimal ensemble weight for the LGBMClassifier as determined during training, computed as 1 - best_a - best_b.
    Output:
        tuple: A tuple containing:
            - y_pred (numpy.ndarray): An array of predicted class labels for the test dataset, derived from the ensemble method.
            - accuracy (float): The accuracy score computed by comparing the ensemble predictions with the true labels.
    Function:
        This function evaluates the performance of the ensemble model on a provided test dataset. It extracts the feature set
        and corresponding true labels from the dataset, computes the prediction probabilities for each classifier, and then
        aggregates these probabilities using the optimal ensemble weights. The final prediction for each sample is determined
        by selecting the class with the highest weighted probability. The function returns both the predicted labels and the
        computed accuracy score.
    """
    rf_model, knn_model, lgb_model = models
    
    X_test = test_data.drop(['label', 'orginal_label'], axis=1)
    y_test = test_data['label']
    
    p1 = rf_model.predict_proba(X_test)
    p2 = knn_model.predict_proba(X_test)
    p3 = lgb_model.predict_proba(X_test)
    
    weighted_p = best_a * p1 + best_b * p2 + best_c * p3
    
    y_pred = np.argmax(weighted_p, axis=1)    
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, accuracy
