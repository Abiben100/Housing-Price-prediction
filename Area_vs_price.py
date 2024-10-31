import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# Scatter plot for area vs price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=housing_data, x='area', y='price', alpha=0.7)
plt.title('Price vs Area')
plt.xlabel('Area (Square Feet)')
plt.ylabel('Price')

plt.show()
