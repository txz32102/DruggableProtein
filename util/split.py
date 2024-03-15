import os
import random
import shutil

# Set the random seed for reproducibility
random.seed(42)

# Source folder containing the files
source_folder = "Tdark"

# Destination folder for the selected files
destination_folder = "Tdark_"

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List all files in the source folder
all_files = os.listdir(source_folder)

# Shuffle the list of files to randomize the selection
random.shuffle(all_files)

# Select 704 random files
selected_files = all_files[:704]

# Move the selected files to the destination folder
for file_name in selected_files:
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)
    shutil.move(source_path, destination_path)

print(
    f"Moved {len(selected_files)} files from '{source_folder}' to '{destination_folder}'."
)
