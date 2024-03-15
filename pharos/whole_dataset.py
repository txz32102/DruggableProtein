# !pip install fair-esm
# from google.colab import drive
# drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


class pharos(Dataset):
    def __init__(self, x, y):
        super(pharos, self).__init__()
        self.data = torch.from_numpy(x).float()
        self.labels = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_labels(self):
        return self.labels

    def get_data(self):
        return self.data


class Cnn(nn.Module):
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

        self.conv_1 = nn.Conv1d(
            kernel_size=self.kernel_1,
            out_channels=self.channel_1,
            in_channels=1,
            stride=1,
        )
        self.normalizer_1 = nn.BatchNorm1d(self.channel_1)
        self.pooling_1 = nn.MaxPool1d(kernel_size=self.kernel_1, stride=stride)

        self.dropout = nn.Dropout(p=drop_out)
        self.fc1 = nn.LazyLinear(64)
        self.normalizer_2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 2)

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


def main():
    if os.path.exists("/content"):
        if os.path.exists("/content/drive/MyDrive"):
            os.chdir("/content/drive/MyDrive")
            df = pd.read_csv("esm2_320_dimensions_with_labels.csv")
        else:
            os.chdir("/home/musong/Desktop")
            df = pd.read_csv("/home/musong/Desktop/esm2_320_dimensions_with_labels.csv")
    else:
        os.chdir("/home/musong/Desktop")
        df = pd.read_csv("/home/musong/Desktop/esm2_320_dimensions_with_labels.csv")

    X = df.drop(["label", "UniProt_id"], axis=1)
    y = df["label"].apply(lambda x: 0 if x != 1 else x)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scalar = MinMaxScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)

    batch_size = 16
    lr = 0.0001
    epochs = 10
    weight_decay = 0

    model = Cnn(output_dim=1, input_dim=320, drop_out=0, stride=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    train_set = pharos(np.array(X_resampled), np.array(y_resampled))
    test_set = pharos(np.array(X_train), np.array(y_test))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        train_predictions = []
        train_labels = []

        # Training phase
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Get class predictions (0, 1, 2, etc.) based on the class with the highest probability
            class_predictions = torch.argmax(outputs, dim=1)

            train_labels += labels.cpu().numpy().tolist()
        model.eval()

        test_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Iterate through the test data using the test_loader
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Perform forward pass and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get class predictions (0, 1, 2, etc.) based on the class with the highest probability
            class_predictions = torch.argmax(outputs, dim=1)

            # Update correct_predictions and total_predictions
            correct_predictions += (class_predictions == labels).sum().item()
            total_predictions += labels.size(0)

        # Calculate test accuracy and other metrics as needed
        test_accuracy = correct_predictions / total_predictions
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


main()
