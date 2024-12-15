#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from pathlib import Path

# Test if in the correct working directory

cwd = Path().resolve()

if not (cwd / "src").is_dir():
    os.chdir(cwd.parent)

# Load the datasets
train_df = pd.read_csv('datasets\datatraining.txt')
test_df = pd.read_csv('datasets\datatest.txt')
test2_df = pd.read_csv('datasets\datatest2.txt')

# Convert date column to datetime
train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])
test2_df['date'] = pd.to_datetime(test2_df['date'])

# Create figure for time series plots
plt.figure(figsize=(20, 12))

# Plot 1: Time series of all sensors
features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
for i, feature in enumerate(features, 1):
    plt.subplot(3, 2, i)
    plt.plot(train_df['date'], train_df[feature], alpha=0.7)
    plt.title(f'{feature} Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Distribution plots for each feature by occupancy
plt.figure(figsize=(20, 12))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 2, i)
    sns.kdeplot(data=train_df, x=feature, hue='Occupancy', common_norm=False)
    plt.title(f'Distribution of {feature} by Occupancy')
    plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Box plots for each feature by occupancy
plt.figure(figsize=(20, 12))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=train_df, x='Occupancy', y=feature)
    plt.title(f'Box Plot of {feature} by Occupancy')
plt.tight_layout()
plt.show()

# Plot 4: Occupancy patterns by hour
train_df['hour'] = train_df['date'].dt.hour
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='hour', hue='Occupancy')
plt.title('Occupancy Patterns by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Count')
plt.show()

# Plot 5: Correlation Heatmap with annotations
plt.figure(figsize=(10, 8))
correlation = train_df[features + ['Occupancy']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# Data preprocessing and statistics
print("\nBasic Statistics for each feature when room is occupied vs unoccupied:")
print("\nOccupied:")
print(train_df[train_df['Occupancy'] == 1][features].describe())
print("\nUnoccupied:")
print(train_df[train_df['Occupancy'] == 0][features].describe())

# Calculate feature importance using standard deviation ratio
std_ratio = {}
for feature in features:
    occupied_std = train_df[train_df['Occupancy'] == 1][feature].std()
    unoccupied_std = train_df[train_df['Occupancy'] == 0][feature].std()
    std_ratio[feature] = abs(occupied_std - unoccupied_std) / min(occupied_std, unoccupied_std)

print("\nFeature Variability Ratio (Higher ratio suggests better discriminative power):")
for feature, ratio in sorted(std_ratio.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {ratio:.2f}")

# Class balance
print("\nClass Balance:")
print(train_df['Occupancy'].value_counts(normalize=True) * 100)