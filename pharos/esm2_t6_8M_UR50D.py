# !pip install fair-esm
# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import torch
from torch.utils.data import Dataset
import esm
import numpy as np
import csv
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import tensorflow as tf

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
    embedded_data = pd.read_csv('output0.csv')
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
    embedded_data.to_csv('embedded_data_with_label.csv', index=False)

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
            df = pd.read_csv('embedded_data_with_label.csv') 
        else:
            # Change to '/home/musong/Desktop' if '/content/drive/MyDrive' doesn't exist
            os.chdir('/home/musong/Desktop')
            df = pd.read_csv('/home/musong/Desktop/embedded_data_with_label.csv') 
    else:
        # Change to '/home/musong/Desktop' if '/content' doesn't exist
        os.chdir('/home/musong/Desktop')
        df = pd.read_csv('/home/musong/Desktop/embedded_data_with_label.csv') 
      
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

def ESM_CNN(X_train, y_train, X_test, y_test):
    from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D,Conv1D
    from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, AveragePooling1D, MaxPooling1D
    from keras.models import Sequential,Model
    from keras.optimizers import SGD
    from keras.callbacks import ModelCheckpoint,LearningRateScheduler, EarlyStopping
    import keras
    from keras import backend as K
    import tensorflow as tf
    inputShape=(320,1)
    input = Input(inputShape)
    x = Conv1D(128,(3),strides = (1),name='layer_conv1',padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D((2), name='MaxPool1',padding="same")(x)
    x = Dropout(0.15)(x)
    x = Conv1D(32,(3),strides = (1),name='layer_conv2',padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D((2), name='MaxPool2',padding="same")(x)
    x = Dropout(0.15)(x)
    x = Flatten()(x)
    x = Dense(64,activation = 'relu',name='fc1')(x)
    x = Dropout(0.15)(x)
    x = Dense(2,activation = 'softmax',name='fc2')(x)
    model = Model(inputs = input,outputs = x,name='Predict')

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
    )

    # Create an optimizer with the learning rate schedule
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    # learning deccay setting
    import math
    def step_decay(epoch): # gradually decrease the learning rate 
        initial_lrate=0.1
        drop=0.6
        epochs_drop = 3.0
        lrate= initial_lrate * math.pow(drop,    # math.pow base raised to a power
            math.floor((1+epoch)/epochs_drop)) # math.floor Round numbers down to the nearest integer
        return lrate
    lrate = LearningRateScheduler(step_decay)

    # early stop setting
    early_stop = EarlyStopping(monitor='val_accuracy', patience = 40,restore_best_weights = True)

    # summary the callbacks_list
    callbacks_list = [ lrate , early_stop]

    model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=200,callbacks=callbacks_list,batch_size = 8, verbose=1)
    return model, model_history

def train(X_train, y_train):
    from sklearn.model_selection import KFold
    k = 10 
    kf = KFold(n_splits=k, shuffle = True, random_state=1)
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train)

    # result collection list
    ACC_collecton = []
    BACC_collecton = []
    Sn_collecton = []
    Sp_collecton = []
    MCC_collecton = []
    AUC_collecton = []

    for train_index , test_index in kf.split(y_train):
        X_train_CV , X_valid_CV = X_train.iloc[train_index,:],X_train.iloc[test_index,:]
        y_train_CV , y_valid_CV = y_train.iloc[train_index] , y_train.iloc[test_index]
        model, model_history = ESM_CNN(X_train_CV, y_train_CV, X_valid_CV, y_valid_CV)
        # confusion matrix 
        predicted_class= []
        predicted_protability = model.predict(X_valid_CV,batch_size=1)
        for i in range(predicted_protability.shape[0]):
            index = np.where(predicted_protability[i] == np.amax(predicted_protability[i]))[0][0]
            predicted_class.append(index)
        predicted_class = np.array(predicted_class)
        y_true = y_valid_CV    
        from sklearn.metrics import confusion_matrix
        import math
        # np.ravel() return a flatten 1D array
        TP, FP, FN, TN = confusion_matrix(y_true, predicted_class).ravel() # shape [ [True-Positive, False-positive], [False-negative, True-negative] ]
        ACC = (TP+TN)/(TP+TN+FP+FN)
        ACC_collecton.append(ACC)
        Sn_collecton.append(TP/(TP+FN))
        Sp_collecton.append(TN/(TN+FP))
        MCC = (TP*TN-FP*FN)/math.pow(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)),0.5)
        MCC_collecton.append(MCC)
        BACC_collecton.append(0.5*TP/(TP+FN)+0.5*TN/(TN+FP))
        from sklearn.metrics import roc_auc_score
        AUC = roc_auc_score(y_valid_CV, predicted_protability[:,1])
        AUC_collecton.append(AUC)
        model.save('train1',save_format = 'tf') 
        # !zip -r /content/AHT_main_tensorflow_model.zip /content/AHT_main_tensorflow_model
        return ACC_collecton[0], BACC_collecton[0], Sn_collecton[0], MCC_collecton[0], AUC_collecton[0]
    

def load_model(model_type):
    # Load the model with saved weights
    loaded_model = tf.keras.models.load_model('model_type')

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
    )

    # Create an optimizer with the learning rate schedule
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    # Compile the loaded model with the same compile settings
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return loaded_model


def test(X_test, y_test):
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
    model = load_model('train1')
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis = 1)
    correct_predictions = np.sum(predicted_classes == y_test)
    total_samples = len(y_test)
    accuracy = correct_predictions / total_samples
    return accuracy

def main():
    X_train, y_train, X_test, y_test = train_and_test()
    train(X_train, y_train)

main()