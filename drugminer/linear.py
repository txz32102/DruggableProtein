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
    confusion_matrix,
    precision_score,
)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

df = pd.read_csv("data/drugminer/esm2_320_dimensions_with_labels.csv")
X = df.drop(["label", "UniProt_id"], axis=1).values
y = df["label"].apply(lambda x: 0 if x != 1 else x).values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(320, 180)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(180, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 30)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x


model = Deep()
model.load_state_dict(torch.load("drugminer/linear.pt"))
with torch.no_grad():
    y = model(X_test).reshape(-1)
    y_predict = y.numpy()
y_test = y_test.reshape(-1).numpy()

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)

tn, fp, fn, tp = confusion_matrix(y_test, y_predict > 0.5).ravel()

# Calculating ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)

# Calculating other metrics
accuracy = accuracy_score(y_test, y_predict > 0.5)
precision = precision_score(y_test, y_predict > 0.5)
sensitivity = recall_score(y_test, y_predict > 0.5)  # Sensitivity is the same as recall
specificity = tn / (tn + fp)
f_score = f1_score(y_test, y_predict > 0.5)
mcc = matthews_corrcoef(y_test, y_predict > 0.5)

# Printing the metrics
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Sensitivity (Recall):", round(sensitivity, 4))
print("Specificity:", round(specificity, 4))
print("F-score:", round(f_score, 4))
print("Matthews Correlation Coefficient (MCC):", round(mcc, 4))
print("ROC AUC:", round(roc_auc, 4))


plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()
