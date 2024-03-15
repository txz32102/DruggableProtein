import torch.nn as nn
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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


class RNN(nn.Module):
    def __init__(
        self, input_dim=320, hidden_dim=64, output_dim=2, num_layers=1, dropout=0
    ):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        x = x.unsqueeze(1)
        # We need to unsqueeze the input tensor to match the batch_first=True setting
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Extract the output of the last time step (many-to-one architecture)
        out = out[:, -1, :].to(device)

        # Pass the output through the fully connected layer
        out = self.fc(out)

        # Apply softmax activation
        out = torch.softmax(out, dim=-1)
        return out

    def last_three(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        x = x.unsqueeze(1)
        # We need to unsqueeze the input tensor to match the batch_first=True setting
        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        return out


class CustomDataset(Dataset):
    def __init__(self, x, y):
        super(CustomDataset, self).__init__()
        self.data = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data


def get_th_dataset(x, y):
    """
    assemble a dataset with the given data and labels
    :param x:
    :param y:
    :return:
    """
    _dataset = CustomDataset(x, y)
    return _dataset


df = pd.read_csv("data/drugminer/esm2_320_dimensions_with_labels.csv")
y = df["label"].apply(lambda x: 0 if x != 1 else x).to_numpy().astype(np.int64)
X = df.drop(["label", "UniProt_id"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)

device = "cpu"
checkpoint = "drugminer/rnn.pt"
checkpoint = torch.load(checkpoint, map_location="cpu")
model = RNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_set = get_th_dataset(X_test, y_test)
with torch.no_grad():
    y_score = model(test_set.get_data())
y_predict = []
for i in range(len(y_score)):
    temp = y_score[i]
    if temp[0] >= 0.5:
        temp_ = 1 - temp[0]
    else:
        temp_ = temp[1]
    y_predict.append(temp_.item())
y_predict = np.array(y_predict)
y_test = test_set.get_labels().cpu().numpy()

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
plt.savefig("debug/CNN.png", dpi=500)


X = model.last_three(test_set.get_data()).detach().numpy()
X_reshaped = X.reshape(2431, 64)
y = test_set.get_labels().numpy()
import seaborn as sns
import umap

umap_reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean"
)
umap_result = umap_reducer.fit_transform(X_reshaped)

plt.figure(figsize=(6, 6))

sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=y_test, palette="viridis")

plt.title("UMAP Projection of the Last Layer Output")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")


plt.savefig("111.svg")
plt.show()
