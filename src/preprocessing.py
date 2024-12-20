import os, random
import pandas as pd

from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------------------------------------------------


def fix_header(input_file):
    
    # Reading the file
    
    with open(input_file, 'r+') as file:
        lines = file.readlines()
        
        # Split the headings line by the first comma delimiter
        
        parts = lines[0].split(',',1)
        
        # Reassign the first part and join
        
        parts[0] = '"ID","Date",'
        lines[0] = parts[0] + parts[1]

        # Writing to file
        
        file.writelines(lines)

    file.close()


def shuffle_and_split(input_file, split_count, first_output_dataset, second_output_dataset):

    # Open the file and read its contents
    
    with open(input_file, 'r') as file:
        lines = file.readlines()
        
    # Set the header line and the lines to be shuffled
    
    vars_header = lines[0]
    data_lines = lines[1:]
    
    # Shuffle the lines
    
    random.shuffle(data_lines)
    
    # Split the lines and make two different halves of the split index
    
    lines_datatest2 = [vars_header] + data_lines[:split_count]
    lines_datavalidation = [vars_header] + data_lines[split_count:]
    
    # Overwrite the old file name
    
    os.rename(input_file, "old_" + input_file)
    
    file.close()

    # Write the each set of lines to their output files
    
    with open(first_output_dataset, 'w') as dataset:
        dataset.writelines(lines_datatest2)
        dataset.close()
    
    with open(second_output_dataset, 'w') as dataset:
        dataset.writelines(lines_datavalidation)
        dataset.close()


def print_distribution(file, test_variable):

    # Read file
    
    df = pd.read_csv(file)
    
    # Find the sum of all 0s and 1s for each file
    
    count_0s = (df[test_variable] == 0).sum()
    count_1s = (df[test_variable] == 1).sum()
    
    # Print the differences
    
    print(f"Distribution of {file}\nTest variable: {test_variable}")
    print(f"Total 0s: {count_0s}\nTotal 1s: {count_1s}")
    
    # Calculate as a ratio
    
    perc_0s = round(((count_0s) / (count_0s + count_1s)) * 100, 2)
    perc_1s = round(((count_1s) / (count_0s + count_1s)) * 100, 2)
    
    print(f"Ratio (True:False) = {perc_1s}:{perc_0s}")


def smote_data(file):
    
    # Reading the file
    
    df = pd.read_csv(file)
    
    # Encoding the non-numeric entry date into a numeric datatype that can be used to SMOTE
    
    label_encoder = LabelEncoder()
    df['Date'] = label_encoder.fit_transform(df['Date'])
    
    # Drop columns

    X = df.drop(columns = 'Occupancy')
    y = df['Occupancy']

    # Resampling
    
    smote = SMOTE(random_state = 42)
    X_res, y_res = smote.fit_resample(X, y)

    # Overwriting the file
    
    df_resampled = pd.concat([X_res, y_res], axis=1)
    df_resampled.columns = df.columns
    df_resampled.to_csv(file, index = False, quoting = 1)


# --------------------------------------------------------------------------------------------


# Test if in the correct working directory, else change current working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)

# Make the new usable directory and make the usable directory

os.chdir(cwd / "datasets")
cwd = Path().resolve()

if not (cwd / "usable").is_dir():
    os.mkdir("usable")
    print("Directory 'usable' created")

# Changing the name of 'datatest.txt'

if not (os.path.exists("datatest1.txt")) and (os.path.exists("datatest.txt")):
    os.rename("datatest.txt", "datatest1.txt")
    print("Dataset datatest.txt has been renamed")

# Declare the file variables

datatraining = ""
datatest1 = ""
datatest2 = ""
datavalidation = ""

# Iterate through all the files in the directory to fix header and assign to the file variables

for file in os.listdir(cwd):
    
    if (file == "datatest1.txt"):
        datatest1 = file
    elif (file == "datatest2.txt"):
        datatest2 = file
    elif (file == "datatraining.txt"):
        datatraining = file
    else:
        continue

    fix_header(file)
    print("Fixed header for file: " + file)

# Count the number of lines in datatest1.txt

dt1_lines = 0

# Minus 1 to exclude the variable header line

with open("datatest1.txt", 'r') as dt1:
    dt1_lines = len(dt1.readlines()) - 1
    dt1.close()

# Validation text file and output files

datavalidation = "datavalidation.txt"
ss_output_file1 = "datatest2.txt"

# Shuffle and split

shuffle_and_split(datatest2, dt1_lines, ss_output_file1, datavalidation)
print("File " + datatest2 + " has been split")

# Testing the balance of the training dataset

print("--------------------------------------------------")

print_distribution(datatraining, "Occupancy")

# Using SMOTE on the training dataset

smote_data(datatraining)

print("--------------------------------------------------")

print_distribution(datatraining, "Occupancy")

print("--------------------------------------------------")

# Moving the usable files to the usable directory

for file in os.listdir(cwd):

    # Testing if its a file or directory

    if (os.path.isdir(file)):
        continue

    # Testing if a file has been marked as "old" and moving files

    parts = ((file.split("."))[0]).split("_")

    if not (len(parts) > 1):
        os.replace(file, "usable/" + file)
        print("Moved file " + file + " to usable directory")
    else:
        continue