import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_curve,
    auc,
    matthews_corrcoef,
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import umap
import umap.plot
import xgboost as xgb
import seaborn as sns

# Load the dataset
df = pd.read_csv(
    "/content/drive/MyDrive/paper/data/drugminer/esm2_320_dimensions_with_labels_original.csv"
)

# Prepare the data
X = df.drop(["label", "UniProt_id"], axis=1)
y = df["label"].apply(lambda x: 0 if x != 1 else x).to_numpy().astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)  # Avoid data leakage


def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_predict > 0.5)
    precision = precision_score(y_test, y_predict > 0.5)
    recall = recall_score(y_test, y_predict > 0.5)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predict > 0.5).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_predict > 0.5)
    mcc = matthews_corrcoef(y_test, y_predict > 0.5)

    return (acc, precision, recall, specificity, f1, mcc), y_predict


# Function to print metrics
def print_metrics(classifier_name, metrics):
    print(f"{classifier_name}")
    print("Accuracy:", metrics[0])
    print("Precision:", metrics[1])
    print("Sensitivity (Recall):", metrics[2])
    print("Specificity:", metrics[3])
    print("F1 Score:", metrics[4])
    print("MCC:", metrics[5])


# Evaluate classifiers
xgb_metrics, xgb_y_predict = evaluate_classifier(
    xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    X_train,
    y_train,
    X_test,
    y_test,
)
print_metrics("XGBoost", xgb_metrics)

nb_metrics, nb_y_predict = evaluate_classifier(
    GaussianNB(), X_train, y_train, X_test, y_test
)
print_metrics("Gaussian Naive Bayes", nb_metrics)

rf_metrics, rf_y_predict = evaluate_classifier(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train,
    y_train,
    X_test,
    y_test,
)
print_metrics("Random Forest", rf_metrics)

svm_metrics, svm_y_predict = evaluate_classifier(
    SVC(probability=True, random_state=42), X_train, y_train, X_test, y_test
)
print_metrics("SVM", svm_metrics)


# Plot ROC curves
def plot_roc_curve(y_test, y_predict, classifier_name, color):
    fpr, tpr, _ = roc_curve(y_test, y_predict)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, color=color, lw=2, label=f"{classifier_name} (AUC = {roc_auc:.2f})"
    )


plt.figure(figsize=(8, 6))
plot_roc_curve(y_test, xgb_y_predict, "XGBoost", "green")
plot_roc_curve(y_test, nb_y_predict, "Gaussian Naive Bayes", "darkorange")
plot_roc_curve(y_test, rf_y_predict, "Random Forest", "blue")
plot_roc_curve(y_test, svm_y_predict, "SVM", "pink")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend(loc="lower right")
plt.show()

# Additional plotting and analysis can be added as required
