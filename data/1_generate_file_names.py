import os
import numpy as np

SPEC_FILE_NAME = "speclist.txt"
NUMS = np.arange(10227, 10242)
HASH_NAMES = [f"v5_13_2_spectra_full_{num}.sha1sum" for num in NUMS]

# Open and clear the speclist.txt file
with open(SPEC_FILE_NAME, "w") as file:
    file.write("")


def write_to_speclist(hash_file_name, spec_file_name):
    """
    Reads a hash file and appends corresponding filenames to the spec file.
    Each line added is in the format: plate_num/filename
    """
    hash_folder = "data/hashes/"
    plate_num = f"{hash_file_name[-13:-8]}/"

    with open(os.path.join(hash_folder, hash_file_name), "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(" ")
        if len(parts) >= 3:
            file_name = parts[2].strip()
            with open(spec_file_name, "a") as spec_file:
                spec_file.write(f"{plate_num}{file_name}\n")


for hash_name in HASH_NAMES:
    write_to_speclist(hash_name, SPEC_FILE_NAME)


def count_files(plate_folder):
    """
    Returns the number of files in the specified plate folder.
    """
    return len([
        name for name in os.listdir(plate_folder)
        if os.path.isfile(os.path.join(plate_folder, name))
    ])


# Count total number of spectra downloaded
total_count = count_files("data/full/10000/")
for i in range(10227, 10242):
    total_count += count_files(f"data/full/{i}/")

print("Total number of spectra downloaded:")
print(total_count)
