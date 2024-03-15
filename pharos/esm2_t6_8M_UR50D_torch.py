# !pip install fair-esm
# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import esm
import numpy as np
import csv
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class pharos(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pandas.DataFrame): Pandas DataFrame containing your data.
            transform (callable, optional): Optional transform to be applied to a sample.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        
        # Extract data and label from the DataFrame
        data = sample['sequence']  # Replace 'data_column_name' with the actual name of your data column
        label = sample['Target Development Level']  # Replace 'label_column_name' with the actual name of your label column
        
        # Convert data and label to PyTorch tensors (you can apply transforms here if needed)
        
        if self.transform:
            UniProt_id = sample['UniProt']
            data = [(UniProt_id, data)]
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model.eval()  # disables dropout for deterministic results
            batch_tokens = batch_converter(data)

            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens[2], repr_layers=[33], return_contacts=True)
            data = results
            if label == 'Tclin':
                label = torch.tensor([1])
            else:
                label = torch.tensor([0])
        
        return data, label

    def Tclin(self):
        if 'Target Development Level' in self.dataframe.columns:
            Tclin_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tclin']
        elif 'label' in self.dataframe.columns:
            Tclin_df = self.dataframe[self.dataframe['label'] == 1]
        return Tclin_df
    
    def Tbio(self):
        if 'Target Development Level' in self.dataframe.columns:
            Tbio_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tbio']
        elif 'label' in self.dataframe.columns:
            Tbio_df = self.dataframe[self.dataframe['label'] == -1]
        return Tbio_df
    
    def Tdark(self):
        if 'Target Development Level' in self.dataframe.columns:
            Tdark_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tdark']
        elif 'label' in self.dataframe.columns:
            Tdark_df = self.dataframe[self.dataframe['label'] == -2]
        return Tdark_df
    
    def Tchem(self):
        if 'Target Development Level' in self.dataframe.columns:
            Tchem_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tchem']
        elif 'label' in self.dataframe.columns:
            Tchem_df = self.dataframe[self.dataframe['label'] == -3]
        return Tchem_df
    
    def sequence_len(self):
        LEN = self.dataframe['SequenceColumn'].apply(lambda x: len(x))
        return LEN
    
    def get_lowest_500_sequences(self):
        # Calculate the length of each sequence
        self.dataframe['SequenceLength'] = self.dataframe['sequence'].apply(lambda x: len(x))
        
        # Sort the DataFrame by SequenceLength in ascending order
        sorted_df = self.dataframe.sort_values(by='SequenceLength', ascending=True)
        
        # Select the lowest 500 sequences
        lowest_500_df = sorted_df.head(500)
        
        # Drop the 'SequenceLength' column if you don't need it in the final DataFrame
        lowest_500_df = lowest_500_df.drop(columns=['SequenceLength'])
        
        # Reset the index
        lowest_500_df = lowest_500_df.reset_index(drop=True)
        
        return lowest_500_df
    
    def vector_for_esm_embedding(self):
        UniProt_id = self.dataframe['UniProt'].to_list()
        sequence = self.dataframe['sequence'].to_list()
        res = []
        for i in range(len(UniProt_id)):
            temp = (UniProt_id[i], sequence[i])
            res.append(temp)
        return res

def esm_embeddings(peptide_sequence_list):
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    ## batch tokens are the embedding results of the whole data set
    batch_tokens = batch_tokens.to(device)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        # Here we export the last layer of the EMS model output as the representation of the peptides
        # model'esm2_t6_8M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)  
    token_representations = results["representations"][6]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append((peptide_sequence_list[i][0], token_representations[i, 1 : tokens_len - 1].mean(0)))
    return sequence_representations

def to_csv(data, filename="output.csv"):
    # Check if the file already exists
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file is empty
        if not file_exists:
            header = ["UniProt_id"] + [str(i) for i in range(1, 321)]
            writer.writerow(header)
        
        for i in range(len(data)):
            file.write(f'{data[i][0]}')
            for j in range(320):
                file.write(f',{data[i][1][j]}')
            file.write('\n')

def min_batch(df_sorted, start_, end_):
    df = df_sorted.iloc[start_:end_]
    df = pd.DataFrame(df)
    data = pharos(df).get_lowest_500_sequences()
    data = pharos(data).vector_for_esm_embedding()
    batch_size = 10
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    with tqdm(total=total_batches, desc="Processing Batches") as pbar:
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            embeddings_data = esm_embeddings(batch_data)
            to_csv(embeddings_data, "output.csv")
            pbar.update(1)
    pbar.clear()


# this main funtion to used to generate data after embedding, note that not all the data is embedded since some sequences are too long for colab
def get_embedded_data():
    if os.path.exists('output.csv'):
        os.remove('output.csv')
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = pd.read_csv('third_merge.csv')
    df['sequence_length'] = df['sequence'].apply(len)
    df_sorted = df.sort_values(by='sequence_length', ascending=True)
    batch_size = 500

    # embedding data is to Q96MM6 after sorted
    for epoch in range((len(df_sorted) - 3000) // batch_size):
        print(f"{epoch + 1} epoch(s)")
        start_ = 0 + batch_size * epoch
        end_ = start_ + batch_size
        df_data = df_sorted.iloc[start_:end_]
        df_data = pd.DataFrame(df_data)
        min_batch(df_data, 0, 500)
    
def label_tackle(df):
    # Check if the '/content' directory exists (for Colab)
    if os.path.exists('/content'):
        # Check if '/content/drive/MyDrive' exists (typical location in Colab)
        if os.path.exists('/content/drive/MyDrive'):
            os.chdir('/content/drive/MyDrive')
        else:
            # Change to '/home/musong/Desktop' if '/content/drive/MyDrive' doesn't exist
            os.chdir('/home/musong/Desktop')
    else:
        # Change to '/home/musong/Desktop' if '/content' doesn't exist
        os.chdir('/home/musong/Desktop')
    embedded_data = pd.read_csv('esm2_320_dimensions.csv')
    label_df = pd.read_csv('paper/raw_data/third_merge.csv')
    label_df['sequence_length'] = label_df['sequence'].apply(len)

    label_df = pd.DataFrame(label_df[['UniProt', 'Target Development Level']])

    # Create a mapping dictionary for the Target Development Level values
    level_mapping = {'Tclin': 1, 'Tdark': -1, 'Tbio': -2}

    # Apply the mapping using the map function
    label_df['Target Development Level'] = label_df['Target Development Level'].map(level_mapping).fillna(-3).astype(int)

    # Convert the DataFrame to a list of lists
    label_df_list = label_df.values.tolist()
    embedded_data_label = pd.DataFrame(embedded_data['UniProt_id'])
    # Convert label_df_list to a dictionary for easy mapping
    label_dict = {row[0]: row[1] for row in label_df_list}

    # Map 'UniProt_id' in embedded_data_label to 'UniProt' in label_df_list
    embedded_data_label['label'] = embedded_data_label['UniProt_id'].map(label_dict)
    embedded_data['label'] = embedded_data_label['label'] 
    embedded_data.to_csv('esm2_320_dimensions_with_labels.csv', index=False)

def balanced_data(df):
    df_Tclin = pharos(df).Tclin()
    df_Tbio = pharos(df).Tbio()
    df_Tchem = pharos(df).Tchem()
    df_Tdark = pharos(df).Tdark()
    train_df = pd.concat([df_Tclin.iloc[0:300], df_Tdark.iloc[0:300]], ignore_index=True)
    test_df = pd.concat([df_Tclin.iloc[300:400], df_Tdark.iloc[300:400]], ignore_index=True)
    return train_df, test_df

def data_fit(train_df, test_df):
    np.random.seed(42)
    X_train = train_df.iloc[:, 1:321]
    y_train = train_df['label']
    y_train[y_train != 1] = 0
    X_test = test_df.iloc[:, 1:321]
    y_test = test_df['label']
    y_test[y_test != 1] = 0
    # Shuffle the training data
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    # Shuffle the test data
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    return X_train, y_train, X_test, y_test

def train_and_test():
    # Check if the '/content' directory exists (for Colab)
    if os.path.exists('/content'):
        # Check if '/content/drive/MyDrive' exists (typical location in Colab)
        if os.path.exists('/content/drive/MyDrive'):
            os.chdir('/content/drive/MyDrive')
            df = pd.read_csv('esm2_320_dimensions_with_labels.csv') 
        else:
            # Change to '/home/musong/Desktop' if '/content/drive/MyDrive' doesn't exist
            os.chdir('/home/musong/Desktop')
            df = pd.read_csv('/home/musong/Desktop/esm2_320_dimensions_with_labels.csv') 
    else:
        # Change to '/home/musong/Desktop' if '/content' doesn't exist
        os.chdir('/home/musong/Desktop')
        df = pd.read_csv('/home/musong/Desktop/esm2_320_dimensions_with_labels.csv') 
      
    train_df, test_df = balanced_data(df)
    X_train, y_train, X_test, y_test = data_fit(train_df, test_df)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train) # normalize X to 0-1 range 
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test



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


def get_dataloader(dataset: Dataset, batch_size):
    """
    assemble a dataloader with the given dataset
    :param dataset:
    :param batch_size:
    :return:
    """
    _dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True,
                             drop_last=True, shuffle=True)
    return _dataLoader

def get_confusion_matrix(y_pred: torch.Tensor, y_test: torch.Tensor):
    """
    plot confusion matrix
    :param y_pred: predictions
    :param y_test: ground truth labels
    :return:
    """
    predictions = torch.argmax(y_pred, dim=-1).numpy()
    labels = torch.argmax(y_test, dim=-1).numpy()  # A:0, B:1, C:2, [D:3]
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(labels))
    disp.plot()
    plt.show()
    return cm



def scores(y_pred: torch.Tensor, y_test: torch.Tensor):
    predictions = torch.argmax(y_pred, dim=-1).numpy()
    labels = y_test.numpy()
    # labels = th.argmax(y_test, dim=-1).numpy()
    recall = recall_score(y_pred=predictions, y_true=labels, average='binary')
    precision = precision_score(y_pred=predictions, y_true=labels, average='binary')
    f1 = f1_score(y_pred=predictions, y_true=labels, average='binary')
    accuracy = accuracy_score(y_pred=predictions, y_true=labels)
    # auc_score = roc_auc_score(y_score=y_pred.detach().numpy(), y_true=y_test.detach().numpy())
    corr = matthews_corrcoef(y_true=labels, y_pred=predictions)
    balanced_accuracy = balanced_accuracy_score(y_true=labels, y_pred=predictions, )

    report = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        # "auc": auc_score,
        'matthews_corrcoef': corr,
        'balanced_accuracy': balanced_accuracy
    }
    return report


def report(model: torch.nn.Module, dataset: CustomDataset):
    _inputs, _labels = dataset.get_data(), dataset.get_labels()
    print(_inputs.size(0))
    predictions = model(_inputs)
    res = scores(predictions, _labels.squeeze())
    print('accuracy ' + str(res["accuracy"]))
    print('precision ' + str(res["precision"]))
    print('f1 ' + str(res["f1"]))
    print('recall ' + str(res["recall"]))
    # print('auc_score ' + str(res["auc"]))
    print('matthews_corrcoef ' + str(res["matthews_corrcoef"]))
    print('balanced_accuracy ' + str(res["balanced_accuracy"]))
    # get_confusion_matrix(predictions, _labels.squeeze())


import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
    

from torch.nn import Module, Conv1d, Linear, Dropout, MaxPool1d, functional as F, BatchNorm1d, LazyLinear


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

        self.conv_1 = Conv1d(kernel_size=self.kernel_1, out_channels=self.channel_1, in_channels=1, stride=1)
        self.normalizer_1 = BatchNorm1d(self.channel_1)
        self.pooling_1 = MaxPool1d(kernel_size=self.kernel_1, stride=stride)

        self.dropout = Dropout(p=drop_out)
        self.fc1 = LazyLinear(64)
        self.normalizer_2 = BatchNorm1d(64)
        self.fc2 = Linear(64, 2)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (batch, embedding_dim) -> (batch, 1, embedding_dim)
        c_1 = self.pooling_1(F.relu(self.normalizer_1(self.conv_1(x))))

        c_2 = torch.flatten(c_1, start_dim=1)
        c_2 = self.dropout(c_2)
        out = F.relu(self.normalizer_2(self.fc1(c_2)))
        out = self.fc2(out)
        out = torch.softmax(out, dim=-1)
        return out
    
from torch.nn import Module
from torch.optim import Optimizer
import os
from tqdm import tqdm


def to_log(epoch: int, loss: float, accuracy, logFile: str, is_append: bool):
    info = str(epoch) + ' ' + str(loss) + ' ' + str(accuracy) + '\n'
    flag = 'a' if is_append else 'w'
    file = open(logFile, flag)  # append mode
    file.write(info)
    file.close()


def train(model: Module, EPOCHS, optimizer: Optimizer, criteria,
           checkpoint, train_set: DataLoader, vali_set: DataLoader, device, LOG_VALIDATION, LOG_TRAIN):
    """
    fine tune the model and save the best model in the checkpoint
    :param LOG_TRAIN:
    :param LOG_VALIDATION:
    :param device:
    :param model: a Cnn or ConvLSTM model
    :param EPOCHS: hyperparameter Epoch
    :param optimizer: pytorch optimizer
    :param criteria: loss function
    :param checkpoint: model checkpoint
    :param train_set: a dataloader
    :param vali_set: a dataloader
    :return: None
    """
    if os.path.exists(LOG_VALIDATION):
        os.remove(LOG_VALIDATION)
    if os.path.exists(LOG_TRAIN):
        os.remove(LOG_TRAIN)
    model = model.to(device)
    min_vali_loss = float("inf")
    for epoch in tqdm(range(EPOCHS)):
        running_loss = 0.0
        train_acc = []
        vali_loss = 0.0
        model.train()
        counter = 0
        for i, (inputs, labels) in enumerate(train_set):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            # outputs = outputs.squeeze()
            loss = criteria(outputs.float(), labels.float().squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_acc.append(scores(outputs.to("cpu"), labels.to("cpu"))["accuracy"])
            counter = i
        model.eval()
        acc = 0
        for j, (vali_inputs, vali_labels) in enumerate(vali_set):
            vali_labels = vali_labels.to(device)
            vali_inputs = vali_inputs.to(device)
            vali_outputs = model(vali_inputs)
            # vali_outputs = vali_outputs.squeeze()
            acc = scores(vali_outputs.to('cpu'), vali_labels.to('cpu'))["accuracy"]
            vali_loss = criteria(vali_outputs.to(device).float(), vali_labels.to(device).float().squeeze())
            if vali_loss < min_vali_loss:
                min_vali_loss = vali_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint)
        avg_loss = running_loss / counter  # loss per batch
        train_accuracy = sum(train_acc) / len(train_acc)
        # print('epoch {} train_loss: {} vali_loss: {} test_acc: {}'
        #       .format(epoch + 1, f'{avg_loss:5f}', f'{vali_loss:5f}', f'{acc: 5f}'))
        # logs
        to_log(epoch, avg_loss, train_accuracy, LOG_TRAIN, True)
        to_log(epoch, vali_loss.item(), acc, LOG_VALIDATION, True)

def label_mapper(label):
    """
    Map group labels to One-hot encoded labels
    :param label: be either "A", or "B" in the binary classification task
    :return: A integer(either 1 or 0)
    """
    return [1] if label == 0 else [0]

def get_mapped_labels(data, labels,):
    labels = np.array([label_mapper(x) for x in labels]).astype(np.float32)
    if len(data) != len(labels):
        raise ValueError("unmatched dataset")
    return labels


def get_dataset_weight(labels: np.ndarray):
    """
    get weights in case of imbalanced classes
    :param labels: labels
    :return: a vector of class weights: ndarray
    """
    weight = compute_class_weight('balanced', classes=np.unique(labels), y=np.squeeze(labels))
    return weight

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    X_train, y_train, X_test, y_test = train_and_test()
    scalar = MinMaxScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)

    plot_range = 10000  # range(1, 10000)
    stratify = True
    batch_size = 16
    lr = 0.0001
    epochs = 500
    weight_decay = 0

    train_loss_log = os.path.join(os.getcwd(), "paper/model/pytorch/train_log.txt")
    test_loss_log = os.path.join(os.getcwd(), "paper/model/pytorch/validation_log.txt")
    checkpoint = os.path.join(os.getcwd(), 'paper/model/pytorch/bestmodel.pt')

    y_train = get_mapped_labels(data=X_train, labels=y_train)
    y_test = get_mapped_labels(data=X_test, labels=y_test)
    weights = get_dataset_weight(y_train)
    train_set = get_th_dataset(X_train, y_train)
    test_set = get_th_dataset(X_test, y_test)
    train_loader = get_dataloader(train_set, batch_size=batch_size)
    test_loader = get_dataloader(test_set, batch_size=len(test_set))

    model = Cnn(output_dim=1, input_dim=320, drop_out=0, stride=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criteria = ASLSingleLabel(gamma_pos=1, gamma_neg=1, eps = 0.1)  # find the best hyperparameter

    train(model=model, EPOCHS=epochs, optimizer=optimizer, checkpoint=checkpoint, criteria=criteria,
        train_set=train_loader, vali_set=test_loader, device=device, LOG_VALIDATION=test_loss_log, LOG_TRAIN=train_loss_log)


    checkpoint = torch.load(checkpoint)
    saved_model = Cnn(output_dim=1, input_dim=320, drop_out=0, stride=2)
    saved_model.load_state_dict(checkpoint['model_state_dict'])
    saved_model.eval()
    report(saved_model, train_set)
    report(saved_model, test_set)