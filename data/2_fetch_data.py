import os

# Define the command
WGET_COMMAND = """#!/bin/bash
wget -nv -r -nH --cut-dirs=7 \\
     -i speclist.txt \\
     -B https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/full/ \\
     -P data/
"""

SCRIPT_FILENAME = "fetch_sdss_data.sh"

# Write the command to a script file
with open(SCRIPT_FILENAME, "w") as script_file:
    script_file.write(WGET_COMMAND)

# Make the script executable
os.chmod(SCRIPT_FILENAME, 0o755)
print(f"Script '{SCRIPT_FILENAME}' has been created and made executable.")

# Uncomment the line below if you want to execute the script automatically
# os.system(f"./{SCRIPT_FILENAME}")