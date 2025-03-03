import numpy as np
import os 

spec_file_name = "speclist.txt"
#nums = np.arange(10227, 10242) 
#nums = np.arange(10239, 10242)
nums =  np.arange(10000, 10001)
hash_names = []

for num in nums:
   hash_names.append("v5_13_2_spectra_full_" + str(num) + ".sha1sum")

# open and clear the speclist.txt file
with open("speclist.txt", "w") as file:
    file.write("")

def write_to_speclist(hash_file_name, spec_file_name):
    hash_folder = "data/hashes/"
#open the sha1sum file and read the lines
    with open(hash_folder + hash_file_name, "r") as file:
        plate_num = str(hash_file_name[-13:-8]) + "/"
        lines = file.readlines()
        for line in lines: 
            #split the line by space
            parts = line.split(" ")
            file_name = parts[2]
            file_name = file_name.replace("\n", "")
            with open(spec_file_name, "a") as file:
                file.write(plate_num + file_name + "\n")

          

for hash_name in hash_names:
   write_to_speclist(hash_name, spec_file_name)

# count num files in full/platenum folder
def count_files(plate_num):
    folder = plate_num
    import os
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])

count = 0
count += (count_files("data/full/"))
for i in range(10227, 10242):
    count += ( count_files("data/full/" + str(i) + "/"))

print("Total number of spectra downloaded: ")
print(count)

