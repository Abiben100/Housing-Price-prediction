import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import t
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# Preprocess the data
for col in housing_data.columns:
    if housing_data[col].dtype == 'object' and housing_data[col].nunique() <= 2:
        housing_data[col] = housing_data[col].map({'yes': 1, 'no': 0})

furnishing_mapping = {'furnished': 3, 'semi-furnished': 2, 'unfurnished': 1}
housing_data['furnishingstatus'] = housing_data['furnishingstatus'].map(furnishing_mapping)
housing_data.fillna(housing_data.mean(), inplace=True)

# Log-transform the target variable 'price'
housing_data['log_price'] = np.log(housing_data['price'])

# Visualize distributions of original and log-transformed 'price'
plt.figure(figsize=(12, 6))

# Original price distribution
plt.subplot(1, 2, 1)
sns.histplot(housing_data['price'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Log-transformed price distribution
plt.subplot(1, 2, 2)
sns.histplot(housing_data['log_price'], kde=True, bins=20, color='orange')
plt.title('Distribution of Log-Transformed Price')
plt.xlabel('Log(Price)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# List of columns to analyze for correlation
columns_with_log = ['log_price', 'area', 'stories', 'bedrooms', 'parking']

# Visualize distributions with KDE
plt.figure(figsize=(18, 12))
for i, col in enumerate(columns_with_log, 1):
    plt.subplot(3, 2, i)
    sns.histplot(housing_data[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col.capitalize()}')
    plt.xlabel(col.capitalize())
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Plot the correlation matrix including log-transformed price
correlation_matrix_log = housing_data[columns_with_log].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix_log, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix (with Log-Transformed Price)')
plt.show()

# Calculate VIF (excluding original 'price' and 'log_price')
X_vif = housing_data.drop(['price', 'log_price'], axis=1)
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print("VIF (excluding 'price' and 'log_price'):")
print(vif_data)


# Function to create a regression summary and calculate MSE and R2
def regression_summary(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and residual variance
    y_pred = model.predict(X_test)
    residual_var = mean_squared_error(y_test, y_pred)

    # Calculate standard error for coefficients
    X_train_const = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept
    XTX_inv = np.linalg.inv(X_train_const.T @ X_train_const)
    se_beta = np.sqrt(np.diag(XTX_inv * residual_var))

    # Calculate t-values and p-values
    t_values = model.coef_ / se_beta[1:]  # Exclude intercept
    p_values = [2 * (1 - t.cdf(abs(t_val), df=len(y_train) - X_train.shape[1] - 1)) for t_val in t_values]

    # Confidence intervals
    ci_lower = model.coef_ - 1.96 * se_beta[1:]
    ci_upper = model.coef_ + 1.96 * se_beta[1:]

    # Create a summary table
    summary_df = pd.DataFrame({
        "coef": model.coef_,
        "std err": se_beta[1:],
        "t": t_values,
        "P>|t|": p_values,
        "[0.025": ci_lower,
        "0.975]": ci_upper
    }, index=X.columns)

    # Calculate MSE and R² for predictions
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return summary_df, model, y_test, y_pred, mse, r2


# Define features (X)
X = housing_data.drop(['price', 'log_price'], axis=1)

# Regression summary for log-transformed price
log_summary, log_model, log_y_test, log_y_pred, log_mse, log_r2 = regression_summary(X, housing_data['log_price'])
print("Summary for Log-Transformed Price:")
print(log_summary)
print(f"\nMSE for Log-Transformed Price: {log_mse}")
print(f"R² for Log-Transformed Price: {log_r2}")

# Regression summary for original price
original_summary, original_model, original_y_test, original_y_pred, original_mse, original_r2 = regression_summary(X, housing_data['price'])
print("\nSummary for Original Price:")
print(original_summary)
print(f"\nMSE for Original Price: {original_mse}")
print(f"R² for Original Price: {original_r2}")

# Visualize predictions for log price
plt.figure(figsize=(12, 6))
plt.scatter(log_y_test, log_y_pred, alpha=0.6, color='orange', edgecolor='k')
plt.title('Predicted vs Actual Log Price')
plt.xlabel('Actual Log Price')
plt.ylabel('Predicted Log Price')
plt.grid()
plt.show()

# Visualize predictions for original price
plt.figure(figsize=(12, 6))
plt.scatter(original_y_test, original_y_pred, alpha=0.6, color='skyblue', edgecolor='k')
plt.title('Predicted vs Actual Original Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.grid()
plt.show()
