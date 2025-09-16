# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("sleep_academic.csv")

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Data info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values if any
df.fillna(df.mean(), inplace=True)

# Exploratory Data Analysis
print("\nStatistical Summary:")
print(df.describe())

# Visualizations

# Sleep hours vs GPA
plt.figure(figsize=(8,5))
sns.scatterplot(x='Sleep_Hours', y='GPA', data=df)
plt.title("Sleep Hours vs GPA")
plt.xlabel("Sleep Hours")
plt.ylabel("GPA")
plt.show()

# Study hours vs GPA
plt.figure(figsize=(8,5))
sns.scatterplot(x='Study_Hours', y='GPA', data=df)
plt.title("Study Hours vs GPA")
plt.xlabel("Study Hours")
plt.ylabel("GPA")
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Prepare data for Machine Learning
X = df[['Sleep_Hours', 'Study_Hours', 'Attendance']]  # Features
y = df['GPA']  # Target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualize Actual vs Predicted GPA
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA")
plt.plot([0, 4], [0, 4], 'r--')  # Reference line
plt.show()
