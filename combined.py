import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of columns you want to create count plots for
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)
columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Set up the figure and axes for subplots
plt.figure(figsize=(15, 12))

# Loop through each column and create a count plot
for i, col in enumerate(columns, 1):
    plt.subplot(3, 3, i)  # 3x3 grid, adjust if needed
    ax = sns.countplot(data=housing_data, x=col)
    plt.title(f'Count Plot for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

for i, col in enumerate(columns, 1):
    ax = plt.subplot(3, 3, i)  # Define each subplot separately
    sns.countplot(data=housing_data, x=col, ax=ax)
    plt.title(f'Count Plot for {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

    # Add percentage annotations on top of each bar
    total = len(housing_data[col])  # Total count for percentage calculation
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'  # Calculate percentage
        x = p.get_x() + p.get_width() / 2  # X position for annotation
        y = p.get_height()  # Y position (top of the bar)
        ax.annotate(percentage, (x, y), ha='center', size=10, xytext=(0, 1), textcoords='offset points')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()



