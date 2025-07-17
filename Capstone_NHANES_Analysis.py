
# 🧠 Capstone Project 1: Working with NumPy Matrices (NHANES Data Analysis)
# 📌 Objective: Analyze the body measurements of adult males and females using Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Load data
male = np.loadtxt('nhanes_adult_male_bmx_2020.csv', delimiter=',', skiprows=1)
female = np.loadtxt('nhanes_adult_female_bmx_2020.csv', delimiter=',', skiprows=1)

# Extract weights
female_weights = female[:, 0]
male_weights = male[:, 0]

# Plot histograms
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.hist(female_weights, bins=30, color='pink', edgecolor='black')
plt.title('Female Weights')
plt.xlim(30, 150)
plt.subplot(2, 1, 2)
plt.hist(male_weights, bins=30, color='skyblue', edgecolor='black')
plt.title('Male Weights')
plt.xlim(30, 150)
plt.tight_layout()
plt.show()

# Boxplot
plt.boxplot([female_weights, male_weights], labels=['Female', 'Male'])
plt.title('Boxplot Comparison of Weights')
plt.ylabel('Weight (kg)')
plt.show()

# Describe function
def describe(data, label):
    print(f"--- {label} ---")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"Std Dev: {np.std(data):.2f}")
    print(f"Skewness: {skew(data):.2f}")
    print(f"Kurtosis: {kurtosis(data):.2f}\n")

describe(female_weights, "Female Weights")
describe(male_weights, "Male Weights")

# Add BMI
female_height_m = female[:, 1] / 100
female_bmi = female[:, 0] / (female_height_m ** 2)
female = np.column_stack((female, female_bmi))

# Standardise female
zfemale = (female - np.mean(female, axis=0)) / np.std(female, axis=0)

# Pairplot
df = pd.DataFrame(zfemale[:, [0, 1, 6, 5, 7]], columns=['Weight', 'Height', 'Waist', 'Hip', 'BMI'])
sns.pairplot(df)
plt.suptitle("Scatterplot Matrix (Standardised)", y=1.02)
plt.show()

# Correlation
print("Pearson correlation:\n", df.corr(method='pearson'))
print("\nSpearman correlation:\n", df.corr(method='spearman'))

# Add waist-to-height and waist-to-hip
female_ratios = np.column_stack((female[:, 6] / female[:, 1], female[:, 6] / female[:, 5]))
male_ratios = np.column_stack((male[:, 6] / male[:, 1], male[:, 6] / male[:, 5]))
female = np.column_stack((female, female_ratios))
male = np.column_stack((male, male_ratios))

# Boxplot of ratios
plt.boxplot([female[:, 8], male[:, 8], female[:, 9], male[:, 9]],
            labels=['F Waist/Height', 'M Waist/Height', 'F Waist/Hip', 'M Waist/Hip'])
plt.title("Waist Ratio Comparisons")
plt.ylabel("Ratio")
plt.grid(True)
plt.show()

# Lowest and Highest BMI
bmi_sorted_indices = np.argsort(female[:, 7])
lowest_bmi = zfemale[bmi_sorted_indices[:5], :]
highest_bmi = zfemale[bmi_sorted_indices[-5:], :]
print("Lowest BMI (Standardised):\n", lowest_bmi)
print("\nHighest BMI (Standardised):\n", highest_bmi)
