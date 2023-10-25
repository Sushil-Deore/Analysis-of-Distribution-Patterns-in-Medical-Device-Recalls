import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

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

def train_and_evaluate_naive_bayes(X_train, y_train, X_test, y_test):
    # Preparing the data
    X_train = prepare_data(X_train)
    X_test = prepare_data(X_test)

    # Applying Min-Max scaling to the training and test data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Computing sample weights for balanced class weights
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Creating and training the Multinomial Naive Bayes model
    multi_NB = MultinomialNB()
    multi_NB.fit(X_train_scaled, y_train, sample_weight=weights)

    # Predicting on the test set
    y_pred = multi_NB.predict(X_test_scaled)

    # Getting value counts
    unique_values, counts = np.unique(y_pred, return_counts=True)
    value_counts = dict(zip(unique_values, counts))

    # Calculating the accuracy on the training and test sets
    training_accuracy = multi_NB.score(X_train_scaled, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Calculating the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Generating a classification report
    report = classification_report(y_test, y_pred)

    class_metrics, accuracy, error = analyze_confusion_matrix(y_test, y_pred)

    return y_pred, value_counts, training_accuracy, test_accuracy, cm, report, class_metrics, accuracy, error

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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calling the train_and_evaluate_naive_bayes function
    y_pred, value_counts, training_accuracy, test_accuracy, cm, report, class_metrics, accuracy, error = train_and_evaluate_naive_bayes(X_train, 
                                                                                                                                        y_train, 
                                                                                                                                        X_test,
                                                                                                                                        y_test)
