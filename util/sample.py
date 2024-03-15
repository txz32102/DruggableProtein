import pandas as pd


def read_fasta(file_path):
    sequences = {"Header": [], "Sequence": []}
    current_header = None
    current_sequence = ""

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # New header found
                if current_header is not None:
                    sequences["Header"].append(current_header)
                    sequences["Sequence"].append(current_sequence)
                current_header = line[1:]
                current_sequence = ""
            else:
                # Continue building the sequence
                current_sequence += line

        # Add the last sequence
        if current_header is not None:
            sequences["Header"].append(current_header)
            sequences["Sequence"].append(current_sequence)

    return pd.DataFrame(sequences)


def extract_label(header):
    # Extract label after the "|" symbol
    parts = header.split("|")
    if len(parts) > 1:
        return parts[1].strip()
    else:
        return None


file_path = "data/pharos/pharos.fasta"
fasta_df = read_fasta(file_path)

fasta_df["Label"] = fasta_df["Header"].apply(extract_label)
tclin_df = fasta_df[fasta_df["Label"] == "Tclin"]
tdark_df = fasta_df[fasta_df["Label"] == "Tdark"]
length_tclin_df = len(tclin_df)

random_tdark_df = tdark_df.sample(n=length_tclin_df, random_state=42)

from sklearn.model_selection import train_test_split
import os

# Assuming tclin_df and tdark_df are already defined

# Define the test size
test_size = 0.2

# Split the positive sequences (Tclin) into train and test sets
tclin_train, tclin_test = train_test_split(
    tclin_df, test_size=test_size, random_state=42
)

# Split the negative sequences (Tdark) into train and test sets
tdark_train, tdark_test = train_test_split(
    random_tdark_df, test_size=test_size, random_state=42
)

# Create folders if they don't exist

train_folder = "data/pharos/fastadata/Train"
test_folder = "data/pharos/fastadata/Independent_Test"

# Create folders if they don't exist
for folder in [train_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)


# Function to extract header before the '|' symbol
def extract_header(identifier):
    return identifier.split("|")[0]


# Function to write sequences to fasta file
def write_fasta(filename, dataframe):
    with open(filename, "w") as file:
        for index, row in dataframe.iterrows():
            header = extract_header(row["Header"])
            file.write(f">{header}\n{row['Sequence']}\n")


# Save the sequences to FASTA files in the train and test folders
write_fasta(os.path.join(train_folder, "positive_train_sequence.fasta"), tclin_train)
write_fasta(os.path.join(test_folder, "positive_test_sequence.fasta"), tclin_test)
write_fasta(os.path.join(train_folder, "negative_train_sequence.fasta"), tdark_train)
write_fasta(os.path.join(test_folder, "negative_test_sequence.fasta"), tdark_test)
