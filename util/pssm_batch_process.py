import subprocess
import os
import time
import argparse


def read_fasta(file_path):
    sequences = {}
    current_header = ""
    current_sequence = ""

    with open(file_path, "r") as fasta_file:
        for line in fasta_file:
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_sequence:
                    sequences[current_header] = current_sequence
                current_header = line[1:]
                current_sequence = ""
            else:
                current_sequence += line
        if current_header and current_sequence:
            sequences[current_header] = current_sequence
    return sequences


def run_psiblast(input_file, output_file, database):
    psiblast_cmd = [
        "psiblast",
        "-db",
        database,
        "-num_iterations",
        "3",
        "-inclusion_ethresh",
        "0.001",
        "-query",
        input_file,
        "-out_ascii_pssm",
        output_file,
    ]

    print("Running PSI-BLAST with the following command:")
    print(" ".join(psiblast_cmd))
    start_time = time.time()
    try:
        subprocess.run(psiblast_cmd, check=True)
        print("PSI-BLAST completed successfully.")
        end_time = time.time()
        running_time = end_time - start_time
        print(f"PSI-BLAST took {running_time:.2f} seconds to complete.")
    except subprocess.CalledProcessError as e:
        print(f"PSI-BLAST failed with error: {e}")


def main():
    # Create a parser object
    parser = argparse.ArgumentParser(description="Your script description here.")

    # Add command line arguments
    parser.add_argument("--data", help="Specify the data option", default="data")
    parser.add_argument("--pssm", help="Specify the pssm option", default="pssm")
    parser.add_argument("--input", help="Specify the input option", default="1.fasta")
    parser.add_argument(
        "--database",
        help="Specify the database option",
        default="/home/musong/Desktop/swissprot",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # fasta data sampled from input_path data
    data_path = args.data
    # pssm matrix path
    pssm_path = args.pssm
    # the whole fasta input data for pssm matrix
    input_path = args.input
    # the database you want to search
    database_path = args.database

    fasta_dict = read_fasta(input_path)
    for key, value in fasta_dict.items():
        input_file = os.path.join(data_path, key) + ".fasta"
        with open(input_file, "w") as fasta_file:
            fasta_file.write(f">{key}\n{value}\n")
        output_file = os.path.join(pssm_path, key + ".pssm")
        run_psiblast(input_file, output_file, database_path)


if __name__ == "__main__":
    main()
