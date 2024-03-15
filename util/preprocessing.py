import pandas as pd
import esm
import numpy as np
import torch
from tqdm import tqdm
import argparse


def esm_embeddings(peptide_sequence_list, model_path = None):
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    if(model_path is not None):
        model.state_dict(torch.load(model_path))
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
        sequence_representations.append(
            (
                peptide_sequence_list[i][0],
                token_representations[i, 1 : tokens_len - 1].mean(0),
            )
        )
    return sequence_representations


def get_positive_peptide_sequence_list(file_path):
    peptide_sequence_list = []

    # Open and read the FASTA file
    with open(file_path, "r") as fasta_file:
        lines = fasta_file.readlines()

    # Initialize variables to store the current identifier and sequence
    current_identifier = None
    current_sequence = ""

    # Iterate through the lines in the file
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line.startswith(">"):  # Check if the line is an identifier
            if current_identifier is not None:
                # If we have a previous sequence, add it to the list
                peptide_sequence_list.append((current_identifier, current_sequence))
            # Extract the protein identifier between the first and second '|'
            current_identifier = line.split("|")[1]
            current_sequence = ""  # Reset the current sequence
        else:
            # Append the line to the current sequence
            current_sequence += line

    # Add the last (identifier, sequence) pair to the list
    if current_identifier is not None:
        peptide_sequence_list.append((current_identifier, current_sequence))
    return peptide_sequence_list


def get_negative_peptide_sequence_list(file_path):
    peptide_sequence_list = []

    # Open and read the FASTA file
    with open(file_path, "r") as fasta_file:
        lines = fasta_file.readlines()

    # Initialize variables to store the current identifier and sequence
    current_identifier = None
    current_sequence = ""

    # Iterate through the lines in the file
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if line.startswith(">"):  # Check if the line is an identifier
            if current_identifier is not None:
                # If we have a previous sequence, add it to the list
                peptide_sequence_list.append((current_identifier, current_sequence))
            # Extract the protein identifier by removing the '>'
            current_identifier = line[1:]
            current_sequence = ""  # Reset the current sequence
        else:
            # Append the line to the current sequence
            current_sequence += line

    # Add the last (identifier, sequence) pair to the list
    if current_identifier is not None:
        peptide_sequence_list.append((current_identifier, current_sequence))
    return peptide_sequence_list

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process peptide sequences.')
    parser.add_argument('--negative_test_path', type=str, required=False, default='data/drugminer/fastadata/Independent_Test/negative_test_sequence.fasta',
                        help='Path to the negative test sequence FASTA file')
    parser.add_argument('--positive_test_path', type=str, required=False, default='data/drugminer/fastadata/Independent_Test/positive_test_sequence.fasta',
                        help='Path to the positive test sequence FASTA file')
    parser.add_argument('--negative_train_path', type=str, required=False, default='data/drugminer/fastadata/Train/negative_train_sequence.fasta',
                        help='Path to the negative train sequence FASTA file')
    parser.add_argument('--positive_train_path', type=str, required=False, default='data/drugminer/fastadata/Train/positive_train_sequence.fasta',
                        help='Path to the positive train sequence FASTA file')
    parser.add_argument('--output_path', type=str, required=False, default='data/drugminer/esm2_320_dimensions_with_labels.csv',
                        help='Path for output CSV file')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='if one use the fine-tuning model or you can use the model trained from scratch')
    return parser.parse_args()


def main():
    args = parse_arguments()
    columns = ["UniProt_id", "sequence", "label", "train or test from original data"]
    df = pd.DataFrame(columns=columns)

    negative_test = get_negative_peptide_sequence_list(args.negative_test_path)
    for id, sequence in negative_test:
        row = [id, sequence, 0, "test"]
        df.loc[len(df)] = row
    negative_train = get_negative_peptide_sequence_list(args.negative_train_path)
    for id, sequence in negative_train:
        row = [id, sequence, 0, "train"]
        df.loc[len(df)] = row
    positive_train = get_positive_peptide_sequence_list(args.positive_train_path)
    for id, sequence in positive_train:
        row = [id, sequence, 1, "train"]
        df.loc[len(df)] = row
    positive_test = get_positive_peptide_sequence_list(args.positive_test_path)
    for id, sequence in positive_test:
        row = [id, sequence, 1, "test"]
        df.loc[len(df)] = row

    df["sequence_length"] = df["sequence"].apply(len)
    sorted_df = df.sort_values(by="sequence_length")
    sorted_df.to_csv("temp.csv", index=False)
    sorted_df = sorted_df[sorted_df["sequence_length"] <= 1000]

    column_headers = ["UniProt_id"] + list(range(1, 321)) + ["label"]
    result_df = pd.DataFrame(columns=column_headers)
    for i in range(0, len(sorted_df)):
        row = sorted_df.iloc[i]
        embeddings = esm_embeddings([(row["UniProt_id"], row["sequence"])], model_path=args.model_path)
        data = (
            [embeddings[0][0]]
            + embeddings[0][1].cpu().numpy().tolist()
            + [row["label"]]
        )
        result_df.loc[i] = data
        torch.cuda.empty_cache()
        if i % 100 == 0:
            tqdm.write(f"Processed {i+1} rows")

    result_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()