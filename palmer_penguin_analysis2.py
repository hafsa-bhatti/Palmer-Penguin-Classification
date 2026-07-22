# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:12:54 2025

@author: Hafsa
"""

"""
Analysis of Palmer Penguins Dataset
Focus: Species Classification and Exploratory Data Analysis
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = sns.load_dataset('penguins')

# Display First and Last 5 rows
print("--- First 5 Rows ---")
print(df.head())
print("\n--- Last 5 Rows ---")
print(df.tail())

# Shape and Data Types
print(f"\nShape: {df.shape}")
print("\nData Types:")
print(df.dtypes)

# Summary of Purpose
"""
Purpose: The dataset contains size measurements for three penguin species.
Variables: species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, 
body_mass_g, and sex.
"""

# Handle Missing Values
print("\nMissing Values:\n", df.isnull().sum())
# Justification: Since the dataset is small and rows with missing values 
# often miss all morphological data, we will drop them to maintain accuracy.
df = df.dropna()

# Handle Duplicates
print(f"\nDuplicate rows found: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Convert Data Types
df['species'] = df['species'].astype('category')
df['island'] = df['island'].astype('category')
df['sex'] = df['sex'].astype('category')

# Univariate Visualization
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='body_mass_g', kde=True, hue='species')
plt.title('Distribution of Body Mass by Species')
plt.show()

# Bivariate Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='flipper_length_mm', y='bill_length_mm', hue='species', style='island')
plt.title('Flipper Length vs Bill Length')
plt.show()

# Machine Learning: Random Forest to predict Species
# Prepare data
le = LabelEncoder()
df_ml = df.copy()
df_ml['sex'] = le.fit_transform(df_ml['sex'])
df_ml['island'] = le.fit_transform(df_ml['island'])

X = df_ml.drop('species', axis=1)
y = df_ml['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Main Story
"""
The Gentoo species is the most distinct, being significantly larger in body mass 
and flipper length. Bill dimensions are the primary separators between 
Adelie and Chinstrap penguins.
"""

# Limitations
"""
The dataset is relatively small (approx 344 rows). It lacks temporal data 
(climate changes over years) which might affect penguin health and population.
"""





