import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# Set up the figure size
plt.figure(figsize=(18, 6))

# Box plot for price vs bedrooms
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
sns.boxplot(data=housing_data, x='bedrooms', y='price')
plt.title('Price Distribution by Number of Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Price')

# Box plot for price vs bathrooms
plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
sns.boxplot(data=housing_data, x='bathrooms', y='price')
plt.title('Price Distribution by Number of Bathrooms')
plt.xlabel('Bathrooms')
plt.ylabel('Price')

# Box plot for price vs stories
plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
sns.boxplot(data=housing_data, x='stories', y='price')
plt.title('Price Distribution by Number of Stories')
plt.xlabel('Stories')
plt.ylabel('Price')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
