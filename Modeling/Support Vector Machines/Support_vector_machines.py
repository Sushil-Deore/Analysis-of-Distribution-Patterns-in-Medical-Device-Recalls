import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_data(dataframe, subset_size=1000, random_state=42):
    data_subset = dataframe.sample(n=subset_size, random_state=random_state)
    X = data_subset.drop('openfda.device_class', axis=1)
    y = data_subset['openfda.device_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel, cost):
    model = SVC(kernel=kernel, C=cost, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Kernel: {kernel}, Cost: {cost}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return model, y_pred

def plot_confusion_matrix(y_test, y_pred, classes, ax, title):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

def main():
    df_openFDA = pd.read_csv("your_data.csv")  # Replace with your actual data file

    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data(df_openFDA)

    kernels = ['linear', 'poly', 'rbf']
    costs = [0.1, 1, 10]

    # Create subplots for confusion matrices
    fig, axes = plt.subplots(nrows=len(kernels), ncols=len(costs), figsize=(15, 12))

    for i, kernel in enumerate(kernels):
        for j, cost in enumerate(costs):
            model, y_pred = train_and_evaluate_svm(X_train_scaled, X_test_scaled, y_train, y_test, kernel, cost)

            # Visualization: Confusion Matrix Heatmap
            classes = sorted(y_test.unique())  # Ensure the order of classes
            plot_confusion_matrix(y_test, y_pred, classes, axes[i, j], f"Kernel: {kernel}, Cost: {cost}")

    plt.tight_layout()
    plt.show()

    # Print accuracies for all combinations
    print("\nAccuracies:")
    for i, kernel in enumerate(kernels):
        for j, cost in enumerate(costs):
            _, y_pred = train_and_evaluate_svm(X_train_scaled, X_test_scaled, y_train, y_test, kernel, cost)
            print(f"Kernel: {kernel}, Cost: {cost}, Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    main()