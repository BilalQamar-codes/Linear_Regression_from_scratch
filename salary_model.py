import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Training function with gradient descent
def train_model(x, y, epochs=500, alpha=0.05):
    n_features = x.shape[1]  # Number of features (1 in this case)
    print(n_features)
    w = np.zeros(n_features)  # Initialize weights
    m = len(x)  # Number of examples

    for epoch in range(epochs):
        total_error = 0
        for i in range(m):
            # Predict using current weights
            prediction = np.dot(w, x[i])

            # Compute the error (difference between predicted and actual)
            error = prediction - y[i]

            # Update weights using gradient descent
            w = w - alpha * error * x[i]

            # Accumulate total error for monitoring
            total_error += error**2

    return w

# Prediction function (vectorized)
def predict(x, w):
    return np.dot(x, w)

# Function to calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to calculate R-squared (R²)
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

# Read in the dataset
dataframe = pd.read_csv("Salary_dataset.csv")

# Extract features and target variable
y = dataframe['Salary'].values  # Convert to NumPy array
X = dataframe['YearsExperience'].values.reshape(-1, 1)  # Reshape to 2D array for single feature

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8, random_state=0)

# Train the model and calculate the weight
w = train_model(X_train, Y_train)
print("The weight of the model is: ", w)

# Predict using the test data
predicted_values = predict(X_test, w)

# Calculate and print Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, predicted_values)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate and print R-squared (R²)
r2 = r_squared(Y_test, predicted_values)
print(f"R-squared (R²): {r2*100}")

# Print predicted vs actual values
for j in range(len(predicted_values)):
    print(f"Predicted: {predicted_values[j]:.2f}, Actual: {Y_test[j]:.2f}")

# Optionally plot the results
# plt.scatter(X_test, Y_test, color='blue', label='Actual')
# plt.scatter(X_test, predicted_values, color='red', label='Predicted')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()
