import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import umap

df = pd.read_csv("data/drugminer/esm2_320_dimensions_with_labels.csv")
X = df.drop(["label", "UniProt_id"], axis=1).values
y = df["label"].apply(lambda x: 0 if x != 1 else x).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


class Deep(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.layer1 = nn.Linear(320, 180)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after first activation
        self.layer2 = nn.Linear(180, 60)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after second activation
        self.layer3 = nn.Linear(60, 30)
        self.act3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)  # Dropout after third activation
        self.output = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.act1(self.layer1(x)))
        x = self.dropout2(self.act2(self.layer2(x)))
        x = self.dropout3(self.act3(self.layer3(x)))
        x = self.sigmoid(self.output(x))
        return x

    def umap_layer3(self, x):
        x = self.dropout1(self.act1(self.layer1(x)))
        x = self.dropout2(self.act2(self.layer2(x)))
        x = self.dropout3(self.act3(self.layer3(x)))
        return x


model = Deep()
model.load_state_dict(torch.load("/home/musong/Desktop/paper/debug/best.pt"))
with torch.no_grad():
    y = model(X_test).reshape(-1)
    y_predict = y.numpy()
y_test = y_test.reshape(-1).numpy()

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = auc(fpr, tpr)

tn, fp, fn, tp = confusion_matrix(y_test, y_predict > 0.5).ravel()

# Calculating various metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
sensitivity = tp / (tp + fn)  # also known as recall
specificity = tn / (tn + fp)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Sensitivity: {:.4f}".format(sensitivity))
print("Specificity: {:.4f}".format(specificity))
print("F1 Score: {:.4f}".format(f1_score))
print("MCC: {:.4f}".format(mcc))


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


X_last_three = model.umap_layer3(X_test).detach().numpy()

umap_reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean"
)
umap_result = umap_reducer.fit_transform(X_last_three)

plt.figure(figsize=(6, 6))

sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=y_test, palette="viridis")

plt.title("UMAP Projection of the Last Layer Output")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.show()
