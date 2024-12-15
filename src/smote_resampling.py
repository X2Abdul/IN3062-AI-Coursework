import pandas as pd
import os
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Test if in the correct working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)
    
# using the SMOTE library on the data and saving it
    
def smote_data(input_file):
    
    # Reading the file
    
    df = pd.read_csv(input_file)
    
    # Encoding the non-numeric entry date into a numeric datatype that can be used to SMOTE
    
    label_encoder = LabelEncoder()
    df['date'] = label_encoder.fit_transform(df['date'])
    
    
    
    X = df.drop(columns = 'Occupancy')
    y = df['Occupancy']
    
    smote = SMOTE(random_state = 42)
    X_res, y_res = smote.fit_resample(X, y)
    
    df_resampled = pd.concat([X_res, y_res], axis=1)
    df_resampled.columns = df.columns
    df_resampled.to_csv(output_file, index = False, quoting = 1)
    
    
input_file = 'datasets\datatraining.txt'

os.mkdir('datasets\\resampled')

output_file = 'datasets\\resampled\datatraining.txt'
smote_data(input_file)