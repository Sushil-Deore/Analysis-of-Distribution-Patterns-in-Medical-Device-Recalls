import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
import numpy as np

def prepare_data(df):
    # Performing one-hot encoding
    RCD_dum = pd.get_dummies(df['root_cause_description'], prefix='RCD', dtype=int)
    Distribution_dum = pd.get_dummies(df['distribution'], prefix='distribution', dtype=int)
    EMP_dum = pd.get_dummies(df['event_month_posted'], prefix='EMP', dtype=int)
    EYP_dum = pd.get_dummies(df['event_year_posted'], prefix='EYP', dtype=int)
    DOW_dum = pd.get_dummies(df['day_of_week_posted'], prefix='DOW', dtype=int)
    EMT_dum = pd.get_dummies(df['event_month_terminated'], prefix='EMT', dtype=int)
    EYT_dum = pd.get_dummies(df['event_year_terminated'], prefix='EYT', dtype=int)
    DOWT_dum = pd.get_dummies(df['day_of_week_terminated'], prefix='DOWT', dtype=int)

    # Concatenating the one-hot encoded data with the original DataFrame
    df = pd.concat([df, RCD_dum, Distribution_dum, EMP_dum, EYP_dum, DOW_dum, EMT_dum, EYT_dum, DOWT_dum], axis=1)

    # Dropping the columns that were one-hot encoded
    df = df.drop(['root_cause_description', 'event_month_posted', 'event_year_posted', 'day_of_week_posted',
                  'event_month_terminated', 'event_year_terminated', 'day_of_week_terminated', 'distribution'], axis=1)

    return df

def train_evaluate_decision_tree(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
    # Training a Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=42, 
                                max_depth=max_depth, 
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, 
                                criterion=criterion)
    
    dt.fit(X_train, y_train)

    # Exporting and displaying the Decision Tree graph
    dot_data = StringIO()
    export_graphviz(dt, 
                    out_file=dot_data, 
                    filled=True, 
                    rounded=False, 
                    feature_names=X_train.columns, 
                    class_names=[str(c) for c in y_train.unique()])
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    image = Image(graph.create_png())

    # Predicting on the training and test sets
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    # Calculating accuracy and confusion matrices
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_confusion = confusion_matrix(y_train, y_train_pred)
    test_confusion = confusion_matrix(y_test, y_test_pred)

    return image, train_accuracy, test_accuracy, train_confusion, test_confusion

def perform_grid_search(X_train, y_train):
    # Setting up parameter grid for Grid Search
    params = {
        "max_depth": [2, 3, 5],
        "min_samples_leaf": [5, 8, 12, 15],
        "criterion": ['gini', 'entropy']
    }
    
    # Performing Grid Search
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=params,
        cv=4,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    
    # Getting the best estimator and its performance
    dt_best = grid_search.best_estimator_
    
    return dt_best

def analyze_confusion_matrix(y_test, y_pred):
    num_classes = len(np.unique(y_test))
    class_metrics = {}

    # Calculating the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    for class_idx in range(num_classes):
        # Extracting the relevant row and column for the current class
        TP_value = cm[class_idx, class_idx]
        FP_value = np.sum(cm[:, class_idx]) - TP_value
        FN_value = np.sum(cm[class_idx, :]) - TP_value
        TN_value = np.sum(cm) - (TP_value + FP_value + FN_value)

        # Appending the calculated values to the respective class dictionaries
        class_metrics[class_idx] = {
            "TP": TP_value,
            "TN": TN_value,
            "FP": FP_value,
            "FN": FN_value
        }

    # Calculating classification accuracy
    total_TP = sum(class_metrics[class_idx]["TP"] for class_idx in range(num_classes))
    total_TN = sum(class_metrics[class_idx]["TN"] for class_idx in range(num_classes))
    total_FP = sum(class_metrics[class_idx]["FP"] for class_idx in range(num_classes))
    total_FN = sum(class_metrics[class_idx]["FN"] for class_idx in range(num_classes))
    accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)

    # Calculating classification error
    error = 1 - accuracy

    return class_metrics, accuracy, error

if __name__ == "__main__":
    # Loading the data from clean_openFDA.csv
    df = pd.read_csv("../clean_openFDA.csv")

    # Defining features and target variable
    X = df.drop('openfda.device_class', axis=1)
    y = df['openfda.device_class']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calling the train_evaluate_decision_tree function
    image, train_accuracy, test_accuracy, train_confusion, test_confusion = train_evaluate_decision_tree(X_train, X_test, y_train, y_test)
    dt_best = perform_grid_search(X_train, y_train)

    # Calling the analyze_confusion_matrix function
    class_metrics, accuracy, error = analyze_confusion_matrix(y_test, test_confusion)





