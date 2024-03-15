from torch.nn import (
    Module,
    Conv1d,
    Linear,
    Dropout,
    MaxPool1d,
    functional as F,
    BatchNorm1d,
    LazyLinear,
)
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


class Cnn(Module):
    """
    CNN model
    """

    def __init__(self, output_dim=1, input_dim=320, drop_out=0, stride=2):
        super(Cnn, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_out = drop_out

        self.kernel_1 = 3
        self.channel_1 = 32

        self.conv_1 = Conv1d(
            kernel_size=self.kernel_1,
            out_channels=self.channel_1,
            in_channels=1,
            stride=1,
            padding=1,
        )
        self.normalizer_1 = BatchNorm1d(self.channel_1)
        self.pooling_1 = MaxPool1d(kernel_size=self.kernel_1, stride=stride)

        self.dropout = Dropout(p=drop_out)
        self.fc1 = LazyLinear(64)
        self.normalizer_2 = BatchNorm1d(64)
        self.fc2 = Linear(64, 2)

    def forward(self, x):
        x = torch.unsqueeze(
            x, dim=1
        )  # (batch, embedding_dim) -> (batch, 1, embedding_dim)
        c_1 = self.pooling_1(F.relu(self.normalizer_1(self.conv_1(x))))

        c_2 = torch.flatten(c_1, start_dim=1)
        c_2 = self.dropout(c_2)
        out = F.relu(self.normalizer_2(self.fc1(c_2)))
        out = self.fc2(out)
        out = torch.softmax(out, dim=-1)
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
    X, y, test_size=0.2, random_state=42
)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)


model = Cnn(output_dim=1, input_dim=320, drop_out=0, stride=2)
model.load_state_dict(torch.load("drugminer/cnn.pt")["model_state_dict"])
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
