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
df = pd.read_csv("data/drugminer/esm2_320_dimensions_with_labels.csv")

# Prepare the data
X = df.drop(["label", "UniProt_id"], axis=1)
y = df["label"].apply(lambda x: 0 if x != 1 else x).to_numpy().astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_classifier.fit(X_train, y_train)
xgb_y_predict = xgb_classifier.predict_proba(X_test)[:, 1]
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_predict)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)

# Calculate performance metrics for XGBoost
acc_xgb = accuracy_score(y_test, xgb_y_predict > 0.5)
mcc_xgb = matthews_corrcoef(y_test, xgb_y_predict > 0.5)
f1_xgb = f1_score(y_test, xgb_y_predict > 0.5)
recall_xgb = recall_score(y_test, xgb_y_predict > 0.5)
precision_xgb = precision_score(y_test, xgb_y_predict > 0.5)
tn, fp, fn, tp = confusion_matrix(y_test, xgb_y_predict > 0.5).ravel()
spe_xgb = tn / (tn + fp)

print("XGBoost")
print("Accuracy:", acc_xgb)
print("MCC:", mcc_xgb)
print("F1 Score:", f1_xgb)
print("Recall:", recall_xgb)
print("Precision:", precision_xgb)
print("Specificity:", spe_xgb)


mapper = umap.UMAP().fit(X_test)
# Train Gaussian Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_y_predict = nb_classifier.predict_proba(X_test)[:, 1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_y_predict)
nb_roc_auc = auc(nb_fpr, nb_tpr)
acc_nb = accuracy_score(y_test, nb_y_predict > 0.5)
mcc_nb = matthews_corrcoef(y_test, nb_y_predict > 0.5)
f1_nb = f1_score(y_test, nb_y_predict > 0.5)
recall_nb = recall_score(y_test, nb_y_predict > 0.5)
tn, fp, fn, tp = confusion_matrix(y_test, nb_y_predict > 0.5).ravel()
spe_nb = tn / (tn + fp)
precision_nb = precision_score(y_test, nb_y_predict > 0.5)

print("GaussianNB")
print("accuracy:", acc_nb)
print("MCC:", mcc_nb)
print("F1 Score:", f1_nb)
print("Recall:", recall_nb)
print("Precision:", precision_nb)
print("Specificity:", spe_nb)

# Train Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_y_predict = rf_classifier.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_predict)
rf_roc_auc = auc(rf_fpr, rf_tpr)
acc_rf = accuracy_score(y_test, rf_y_predict > 0.5)
mcc_rf = matthews_corrcoef(y_test, rf_y_predict > 0.5)
f1_rf = f1_score(y_test, rf_y_predict > 0.5)
recall_rf = recall_score(y_test, rf_y_predict > 0.5)
tn, fp, fn, tp = confusion_matrix(y_test, rf_y_predict > 0.5).ravel()
spe_rf = tn / (tn + fp)
precision_rf = precision_score(y_test, rf_y_predict > 0.5)

print("Random Forest")
print("accuracy:", acc_rf)
print("MCC:", mcc_rf)
print("F1 Score:", f1_rf)
print("Recall:", recall_rf)
print("Precision:", precision_rf)
print("Specificity:", spe_rf)


# Train Support Vector Machine (SVM)
svm_classifier = SVC(probability=True, random_state=42)
svm_classifier.fit(X_train, y_train)
svm_y_predict = svm_classifier.predict_proba(X_test)[:, 1]
svm_fpr, svm_tpr, thresholds = roc_curve(y_test, svm_y_predict)
svm_roc_auc = auc(svm_fpr, svm_tpr)
acc_svm = accuracy_score(y_test, svm_y_predict > 0.5)
mcc = matthews_corrcoef(y_test, svm_y_predict > 0.5)
f1 = f1_score(y_test, svm_y_predict > 0.5)
recall = recall_score(y_test, svm_y_predict > 0.5)
tn, fp, fn, tp = confusion_matrix(y_test, svm_y_predict > 0.5).ravel()
spe_svm = tn / (tn + fp)
precision_svm = precision_score(y_test, svm_y_predict > 0.5)


print("SVM")
print("accuracy:", acc_svm)
print("MCC:", mcc)
print("F1 Score:", f1)
print("Recall:", recall)
print("Precision:", precision_svm)
print("Specificity:", spe_svm)

# Create a new figure for both ROC curves
plt.figure(figsize=(8, 6))

# Plot Gaussian Naive Bayes ROC curve
plt.plot(
    nb_fpr,
    nb_tpr,
    color="darkorange",
    lw=2,
    label=f"Gaussian Naive Bayes (AUC = {nb_roc_auc:.2f})",
)

# Plot Random Forest ROC curve
plt.plot(
    rf_fpr, rf_tpr, color="blue", lw=2, label=f"Random Forest (AUC = {rf_roc_auc:.2f})"
)

# Plot SVM ROC curve
plt.plot(svm_fpr, svm_tpr, color="pink", lw=2, label=f"SVM (AUC = {svm_roc_auc:.2f})")

plt.plot(xgb_fpr, xgb_tpr, color="green", lw=2, label=f"XGB (AUC = {svm_roc_auc:.2f})")

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

# Configure the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend(loc="lower right")

# Save the figure
plt.savefig("debug/ML.png", dpi=500)


# Assuming 'mapper' and the prediction variables are defined earlier in your code
# (svm_y_predict, rf_y_predict, nb_y_predict, xgb_y_predict)

# Setting the Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Creating the subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

svm_y_predict = (svm_y_predict >= 0.5).astype(int)
rf_y_predict = (rf_y_predict >= 0.5).astype(int)
nb_y_predict = (nb_y_predict >= 0.5).astype(int)
xgb_y_predict = (xgb_y_predict >= 0.5).astype(int)

umap.plot.points(mapper, labels=svm_y_predict, ax=axs[0, 0], theme="fire")
axs[0, 0].set_title("SVM Classifier")
umap.plot.points(mapper, labels=rf_y_predict, ax=axs[0, 1], theme="fire")
axs[0, 1].set_title("Random Forest Classifier")
umap.plot.points(mapper, labels=nb_y_predict, ax=axs[1, 0], theme="fire")
axs[1, 0].set_title("Naive Bayes Classifier")
umap.plot.points(mapper, labels=xgb_y_predict, ax=axs[1, 1], theme="fire")
axs[1, 1].set_title("XGBoost Classifier")

# Adjusting layout for better spacing
plt.tight_layout()

# Saving the plot
plt.savefig("debug/umap_plot.png", dpi=500)

# Displaying the plot
plt.show()


# ... plot ROC curves ...
# Add XGBoost ROC curve
plt.plot(
    xgb_fpr, xgb_tpr, color="green", lw=2, label=f"XGBoost (AUC = {xgb_roc_auc:.2f})"
)

# ... rest of your plotting code ...

plt.legend(loc="lower right")
plt.savefig("debug/xgb.png", dpi=500)
plt.show()


import csv

# Create a list of dictionaries containing classifier names, FPR, and TPR data
roc_data = [
    {"Classifier": "Gaussian Naive Bayes", "FPR": nb_fpr, "TPR": nb_tpr},
    {"Classifier": "Random Forest", "FPR": rf_fpr, "TPR": rf_tpr},
    {"Classifier": "SVM", "FPR": svm_fpr, "TPR": svm_tpr},
    {"Classifier": "XGB", "FPR": xgb_fpr, "TPR": xgb_tpr},
]


output_file = "matlab/drugminer_roc_data.txt"
with open(output_file, "w") as file:
    # Iterate over the ROC data
    for entry in roc_data:
        classifier_name = entry["Classifier"].replace(" ", "_")
        fpr = entry["FPR"]
        tpr = entry["TPR"]

        # Convert the FPR numpy array to a string with square brackets
        fpr_str = "[" + ", ".join(map(str, fpr)) + "];"
        tpr_str = "[" + ", ".join(map(str, tpr)) + "];"
        # Write the FPR and TPR values to the file in the desired format
        file.write(
            f"{classifier_name}_FPR = {fpr_str}\n{classifier_name}_TPR = {tpr_str}\n"
        )

print(f"ROC data has been written to {output_file}")


# Specify the file name


# Write the data to the CSV file
with open(output_file, mode="w", newline="") as file:
    fieldnames = ["Classifier", "FPR", "TPR"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for data in roc_data:
        writer.writerow(data)
