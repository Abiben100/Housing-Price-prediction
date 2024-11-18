import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# Convert 'yes'/'no' columns to numeric (1/0)
for col in housing_data.columns:
    if housing_data[col].dtype == 'object' and housing_data[col].nunique() <= 2:
        housing_data[col] = housing_data[col].map({'yes': 1, 'no': 0})

# Map 'furnishingstatus' column into 1,2,3 depending on the status
furnishing_mapping = {'furnished': 3, 'semi-furnished': 2, 'unfurnished': 1}
housing_data['furnishingstatus'] = housing_data['furnishingstatus'].map(furnishing_mapping)

# Handle missing values for numerical columns by filling with the column mean
housing_data.fillna(housing_data.mean(), inplace=True)

# Verifying if the missing values are resolved
print("Missing values per column:")
print(housing_data.isnull().sum())

# List of columns to analyze
columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Visualize distributions with KDE
plt.figure(figsize=(18, 12))
for i, col in enumerate(columns, 1):
    plt.subplot(3, 2, i)
    sns.histplot(housing_data[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot the correlation matrix
correlation_matrix = housing_data[columns].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.show()

# Calculate VIF
X = housing_data[columns]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("VIF :")
print(vif_data)

# Define the features and target variable
X = housing_data.drop('price', axis=1)  # Drop 'price' from features
y = housing_data['price']  # Target variable

# Confirm no missing values in X or y
print("Missing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
