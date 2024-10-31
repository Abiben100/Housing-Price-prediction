import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# Set up the figure size
plt.figure(figsize=(14, 6))

# Box plot for price vs furnishingstatus
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
sns.boxplot(data=housing_data, x='furnishingstatus', y='price')
plt.title('Price Distribution by Furnishing Status')
plt.xlabel('Furnishing Status')
plt.ylabel('Price')

# Box plot for price vs prefarea
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
sns.boxplot(data=housing_data, x='prefarea', y='price')
plt.title('Price Distribution by Preferred Area')
plt.xlabel('Preferred Area')
plt.ylabel('Price')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()