import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# Function to create a pie chart for a specified column
def plot_pie_chart(housing_data, prefarea):
    plt.figure(figsize=(6, 6))
    housing_data[prefarea].value_counts().plot.pie(autopct='%1.1f%%')
    plt.ylabel('')  # Remove the y-label for a cleaner look
    plt.title(f'Distribution of {prefarea.capitalize()}')
    plt.show()

# Plotting the pie chart for 'prefarea'
plot_pie_chart(housing_data, 'prefarea')
