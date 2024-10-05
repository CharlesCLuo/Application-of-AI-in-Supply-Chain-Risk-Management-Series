# Import the necessary libraries for data manipulation, numerical operations, visualization, and performance evaluation.
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # For Exponential Moving Average (EMA) modeling.
import matplotlib.pyplot as plt  # For creating visualizations and charts.
from sklearn.preprocessing import MinMaxScaler  # For normalizing and rescaling data.
from sklearn.metrics import mean_absolute_error, mean_squared_error  # For calculating performance metrics.
import numpy as np  # For numerical operations and random number generation.

# Assume 'data_single' and 'demand_scaler' are already defined as in the previous simulation code.

# Split the dataset into training and testing sets.
# We allocate 80% of the data to training and 20% to testing to ensure the model is trained on most of the data
# and tested on the remaining to evaluate its performance.
train_size = int(len(data_single) * 0.8)  # Calculate the number of rows to include in the training set (80% of the total data).
train, test = data_single.iloc[:train_size], data_single.iloc[train_size:]  # Use slicing to separate training and testing data.

# Step 1: Fit the Exponential Moving Average (EMA) model on the training data.
# We use the ExponentialSmoothing function from the statsmodels library.
# Here, we only include a trend component (additive trend) and no seasonal component (set to None).
ema_model = ExponentialSmoothing(train['Demand'], trend='add', seasonal=None).fit()  # Fit the EMA model on the training data.

# Step 2: Forecast using the EMA model.
# We forecast values for the training period using fitted values and predict the next steps for the test period.
train['EMA_Forecast'] = ema_model.fittedvalues  # Use fitted values to get the model's predictions on the training data.
test['EMA_Forecast'] = ema_model.forecast(steps=len(test))  # Forecast future values for the test set.

# Step 3: Rescale predictions and actual values back to the original scale using the demand scaler.
# The 'Demand' and 'EMA_Forecast' columns are currently normalized between 0 and 1, so we need to transform them back to their original scale.
train['Demand_Rescaled'] = demand_scaler.inverse_transform(train[['Demand']])  # Convert the normalized demand back to its original scale.
test['Demand_Rescaled'] = demand_scaler.inverse_transform(test[['Demand']])  # Rescale the test demand data.
# Rescale the EMA forecast values for the test set back to the original scale for comparison.
test['EMA_Forecast_Rescaled'] = demand_scaler.inverse_transform(test[['EMA_Forecast']].values.reshape(-1, 1))

# Step 4: Visualize the EMA results.
plt.figure(figsize=(14, 7))  # Set the figure size to 14 inches by 7 inches for better readability.

# Plot the rescaled actual demand values for the training period.
plt.plot(train.index, train['Demand_Rescaled'], label='Training Data (Rescaled)', color='blue', linestyle='-')  # Training data in blue.
# Plot the rescaled actual demand values for the testing period.
plt.plot(test.index, test['Demand_Rescaled'], label='Test Data (Rescaled)', color='orange', linestyle='--')  # Test data in orange.
# Plot the rescaled EMA forecast values for the testing period.
plt.plot(test.index, test['EMA_Forecast_Rescaled'], label='EMA Forecast', color='red', linestyle='-')  # EMA forecast in red.

# Highlight the point where the training data ends and the test data begins.
plt.axvline(x=train.index[-1], color='black', linestyle=':', linewidth=1.5, label='Train-Test Split')  # Vertical line indicating the split.

# Add labels, title, and legend to make the plot easier to understand.
plt.title('Exponential Moving Average (EMA) Model - Actual vs. Forecasted Demand')  # Title of the plot.
plt.xlabel('Date')  # X-axis label indicating the dates.
plt.ylabel('Demand')  # Y-axis label indicating demand values.
plt.legend()  # Display the legend to identify each line in the plot.
plt.grid(True)  # Add a grid for easier interpretation of the values.

# Display the plot.
plt.show()  # Show the visualization.

# Step 5: Calculate Performance Metrics for the EMA Model
# Calculate Mean Absolute Error (MAE) to measure the average magnitude of the errors between actual and forecasted values.
mae_EMA = mean_absolute_error(test['Demand_Rescaled'], test['EMA_Forecast_Rescaled'])
print(f"Mean Absolute Error (MAE): {mae_EMA:.4f}")

# Calculate Root Mean Squared Error (RMSE) to measure the square root of the average of squared differences between actual and forecasted values.
# This metric penalizes larger errors more than MAE.
rmse_EMA = mean_squared_error(test['Demand_Rescaled'], test['EMA_Forecast_Rescaled'], squared=False)  # Set squared=False to get RMSE.
print(f"Root Mean Squared Error (RMSE): {rmse_EMA:.4f}")

# Calculate Mean Absolute Percentage Error (MAPE) to express the error as a percentage of the actual values.
# Adding a small epsilon value (1e-10) helps avoid division by zero if any demand values are zero.
epsilon = 1e-10  # A small value to handle division by zero, if any demand values are zero.
mape_EMA = np.mean(np.abs((test['Demand_Rescaled'] - test['EMA_Forecast_Rescaled']) / (test['Demand_Rescaled'] + epsilon))) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape_EMA:.4f}%")
