import random
import os
from pathlib import Path

# Test if in the correct working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)

# Shuffle and split

def shuffle_and_split(filepath, split_count, datatest2_output, datavalidation_output):
    
    # Open the file and read its contents
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
    # Set the header line and the lines to be shuffled
    
    vars_header = lines[0]
    data_lines = lines[1:]
    
    # Shuffle the lines
    
    random.shuffle(data_lines)
    
    # Split the lines and make two different halves of the split index
    
    lines_datatest2 = [vars_header] + data_lines[:split_count]
    lines_datavalidation = [vars_header] + data_lines[split_count:]
    
    # Write the each set of lines to their output files
    
    with open(datatest2_output, 'w') as datatest2:
        datatest2.writelines(lines_datatest2)
        print("New datatest2 file created")
    
    with open(datavalidation_output, 'w') as datavalidation:
        datavalidation.writelines(lines_datavalidation)
        print("New datavalidation file created")
       
# Split count = number of lines for datatest.txt

split_count = 2665
        
# Files

input_file = 'datasets\datatest2.txt'
file_datatest2_new = 'datasets\datatest2_new.txt'
file_datavalidation = 'datasets\datavalidation.txt'

shuffle_and_split(input_file, split_count, file_datatest2_new, file_datavalidation)