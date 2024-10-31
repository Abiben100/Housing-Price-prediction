import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# List of columns to plot
columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Set up the figure size and layout
plt.figure(figsize=(18, 12))

# Loop through each column and create a histogram with KDE
for i, col in enumerate(columns, 1):
    plt.subplot(3, 2, i)  # Create a 3x2 grid for subplots
    sns.histplot(housing_data[col], kde=True, bins=20, color='skyblue')  # Histogram with KDE
    plt.title(f'Distribution of {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Calculate the correlation matrix
correlation_matrix = housing_data[columns].corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.show()

# Create a DataFrame for VIF
X = housing_data[columns]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
