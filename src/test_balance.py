import pandas as pd
import os
from pathlib import Path

# Test if in the correct working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)

# Visualise ratio
# Testing for the difference in Occupancy values

def visualise_distribution(filepath, test_class = "Occupancy"):
    
    # Read file
    
    df = pd.read_csv(filepath)
    
    # Find the sum of all 0s and 1s for each file
    
    count_0s = (df[test_class] == 0).sum()
    count_1s = (df[test_class] == 1).sum()
    
    # Print the differences
    
    print(f"{filepath} {test_class} Distribution:")
    print(f"Total 0s: {count_0s}")
    print(f"Total 1s: {count_1s}")
    
    # Calculate as a ratio
    
    perc_0s = round(((count_0s) / (count_0s + count_1s)) * 100, 2)
    perc_1s = round(((count_1s) / (count_0s + count_1s)) * 100, 2)
    
    print(f"Ratio (True:False) = {perc_1s}:{perc_0s}\n")
    

# Files

datatest_file = 'datasets\datatest.txt'
datatest2_file = 'datasets\datatest2_new.txt'
datatraining_file = 'datasets\datatraining.txt'
datavalidation_file = 'datasets\datavalidation.txt'

visualise_distribution(datatest_file)
visualise_distribution(datatest2_file)
visualise_distribution(datatraining_file)
visualise_distribution(datavalidation_file)