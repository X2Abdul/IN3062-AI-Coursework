import os
from pathlib import Path

# Test if in the correct working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)
    
# Fixing the header of the file 

def fix_header(input_file):
    
    # Reading the file
    
    with open(input_file, 'r') as file:
        lines = file.readlines()
        
        # Split the headings line by the first comma delimiter
        
        parts = lines[0].split(',',1)
        
        # Reassign the first part and join
        
        parts[0] = '"ID","Date",'
        lines[0] = parts[0] + parts[1]
        
    # Writing to the file  
        
    with open(input_file, 'w') as file:
        file.writelines(lines)
        
# Files
        
input_file = 'datasets\datatest.txt'
fix_header(input_file)