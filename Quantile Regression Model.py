import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'D:/One Drive/OneDrive/Desktop/Project/Data set/Housing.csv'
housing_data = pd.read_csv(file_path)

# Convert 'yes'/'no' columns to numeric (1/0)
for col in housing_data.columns:
    if housing_data[col].dtype == 'object' and housing_data[col].nunique() <= 2:
        housing_data[col] = housing_data[col].map({'yes': 1, 'no': 0})

# Map 'furnishingstatus' column into 1, 2, 3 depending on the status
furnishing_mapping = {'furnished': 3, 'semi-furnished': 2, 'unfurnished': 1}
housing_data['furnishingstatus'] = housing_data['furnishingstatus'].map(furnishing_mapping)

# Handle missing values for numerical columns by filling with the column mean
housing_data.fillna(housing_data.mean(), inplace=True)

# Log-transform the target variable 'price'
housing_data['log_price'] = np.log(housing_data['price'])

# Define the features and log-transformed target variable (excluding 'bedrooms')
X = housing_data.drop(['price', 'log_price'], axis=1)  # Features (independent variables)
y = housing_data['log_price']  # Log-transformed target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Quantile Regression ---
quantiles = [0.25, 0.5, 0.75]  # Specify quantiles for analysis

# Initialize a dictionary to store results for each quantile
results = {}

# Train and evaluate Quantile Regression models for each quantile
for q in quantiles:
    model_qr = QuantReg(y_train, X_train)
    result = model_qr.fit(q=q)
    results[q] = result  # Store result for later use

    # Make predictions on the test set
    y_pred_q = result.predict(X_test)

    # Evaluate the model on the log-transformed scale
    mse_log = mean_squared_error(y_test, y_pred_q)
    r2_log = r2_score(y_test, y_pred_q)

    # Convert predictions back to the original scale
    y_pred_original = np.exp(y_pred_q)
    mse_original = mean_squared_error(np.exp(y_test), y_pred_original)
    r2_original = r2_score(np.exp(y_test), y_pred_original)

    # Print metrics for Log-Transformed and Original Price
    print(f"\nQuantile: {q}")
    print(result.summary())  # Display summary of the regression for the quantile
    print(f"Log-Transformed MSE for Quantile {q}: {mse_log}")
    print(f"Log-Transformed R2 for Quantile {q}: {r2_log}")
    print(f"Original Scale MSE for Quantile {q}: {mse_original}")
    print(f"Original Scale R2 for Quantile {q}: {r2_original}")

    # Create a custom summary table for original price (exp of the coefficients)
    coef_original = np.exp(result.params)  # Coefficients on original price scale
    summary_original = pd.DataFrame({
        'coef': coef_original,
        'std err': np.exp(result.bse),  # Standard error on original scale
        't': result.tvalues,
        'P>|t|': result.pvalues,
        '[0.025': np.exp(result.conf_int()[0]),  # Lower 95% CI
        '0.975]': np.exp(result.conf_int()[1])  # Upper 95% CI
    })
    print(f"\nOriginal Price Summary for Quantile {q}:")
    print(summary_original)

    # Plot Actual vs. Predicted Values for original price
    plt.figure(figsize=(8, 6))
    plt.scatter(np.exp(y_test), y_pred_original, alpha=0.7, color='blue')
    plt.plot([np.exp(y_test).min(), np.exp(y_test).max()],
             [np.exp(y_test).min(), np.exp(y_test).max()],
             'r--', lw=2, label="Perfect Fit")
    plt.title(f'Quantile {q} Regression: Actual vs Predicted')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



