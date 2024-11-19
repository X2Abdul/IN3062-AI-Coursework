import pandas as pd
import os
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Test if in the correct working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)
    
def smote_data(filepath):
    df = pd.read_csv(filepath)
    label_encoder = LabelEncoder()
    
    df['date'] = label_encoder.fit_transform(df['date'])
    
    # print(df.info())
    
    X = df.drop(columns = 'Occupancy')
    y = df['Occupancy']
    
    smote = SMOTE(random_state = 42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # print(f"Before SMOTE:\n{y.value_counts()}")
    # print(f"After SMOTE:\n{y_res.value_counts()}")
    
    df_resampled = pd.concat([X_res, y_res], axis=1)
    df_resampled.to_csv('datasets\\resampled\datatraining.txt', sep=',', index=False, quoting=pd.io.common.csv.QUOTE_ALL)
    
    
filepath = 'datasets\datatestCOPY.txt'
smote_data(filepath)