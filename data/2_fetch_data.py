import os

# Define the command
wget_command = """#!/bin/bash
wget -nv -r -nH --cut-dirs=7 \\
     -i speclist.txt \\
     -B https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/full/ \\
     -P data/
"""

script_filename = "fetch_sdss_data.sh"
with open(script_filename, "w") as script_file:
    script_file.write(wget_command)

# Make the script executable
os.chmod(script_filename, 0o755)
print(f"Script '{script_filename}' has been created and made executable.")

# Uncomment the line below if you want to execute the script automatically
os.system(f"./{script_filename}")
