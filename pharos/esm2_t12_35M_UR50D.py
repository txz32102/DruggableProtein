import pandas as pd
import torch
from torch.utils.data import Dataset
import esm
import numpy as np

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
        Tclin_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tclin']
        return Tclin_df
    
    def Tbio(self):
        Tbio_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tbio']
        return Tbio_df
    
    def Tdark(self):
        Tdark_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tdark']
        return Tdark_df
    
    def Tchem(self):
        Tchem_df = self.dataframe[self.dataframe['Target Development Level'] == 'Tchem']
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


def main():
    df = pd.read_csv('/kaggle/input/my-test/third_merge.csv')
    df['sequence_length'] = df['sequence'].apply(len)
    df_sorted = df.sort_values(by='sequence_length', ascending=False)
    df = df_sorted.iloc[3000:3100]

    data = pharos(df).get_lowest_500_sequences()
    data = pharos(data).get_lowest_500_sequences()
    data = pharos(data).vector_for_esm_embedding()

    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = data[1:3]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
