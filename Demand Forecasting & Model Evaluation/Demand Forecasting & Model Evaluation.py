# Import necessary libraries
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to prepare data by removing outliers
def prepare_data(df):
    """
    Prepares data by renaming columns, removing outliers based on the IQR method,
    and ensuring no zero values in 'y'.
    
    Parameters:
    df (DataFrame): Input data with columns 'Date', 'Quantity', and 'Material No.'
    
    Returns:
    DataFrame: Cleaned data with renamed columns and outliers removed
    """
    # Select necessary columns and rename for Prophet model
    df = df[['Date', 'Quantity', 'Material No.']]
    df.rename(columns={'Date': 'ds', 'Quantity': 'y'}, inplace=True)
    
    # Remove outliers using the Interquartile Range (IQR) method
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['y'] >= lower_bound) & (df['y'] <= upper_bound)]
    
    # Remove zero values to prevent forecasting errors
    if df['y'].max() == 0:
        df = df[df['y'] > 0]

    return df

# Function to check if data has sufficient variability
def check_data_variability(data):
    """
    Checks if data has enough variability for time series analysis.
    Raises an error if all values are identical.

    Parameters:
    data (DataFrame): Time series data for one material
    """
    if len(data['y'].unique()) <= 1:
        raise ValueError("Input data has insufficient variability")

# Forecasting function using SARIMAX model
def forecast_sarimax(train, periods=12):
    """
    Forecasts future values using the SARIMAX model with seasonal order.

    Parameters:
    train (DataFrame): Training data for SARIMAX model
    periods (int): Number of forecast periods

    Returns:
    np.array: Forecasted values
    """
    check_data_variability(train)  # Ensure data has variability

    # Train SARIMAX model with seasonal parameters
    model = SARIMAX(train['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=periods)
    return np.clip(forecast, a_min=0, a_max=None).round()  # Clip negative values to zero

# Forecasting function using XGBoost model
def forecast_xgboost(train, periods=12):
    """
    Forecasts future values using the XGBoost regression model.

    Parameters:
    train (DataFrame): Training data for XGBoost model
    periods (int): Number of forecast periods

    Returns:
    np.array: Forecasted values
    """
    X = np.arange(len(train)).reshape(-1, 1)
    y = train['y'].values
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(X, y)
    X_forecast = np.arange(len(train), len(train) + periods).reshape(-1, 1)
    forecast = model.predict(X_forecast)
    return np.clip(forecast, a_min=0, a_max=None).round()  # Clip negative values to zero

# Forecasting function using moving average
def forecast_moving_average(train, periods=12):
    """
    Forecasts future values using a simple moving average method.

    Parameters:
    train (DataFrame): Training data
    periods (int): Number of forecast periods

    Returns:
    np.array: Forecasted values based on the moving average
    """
    window = 3  # Define moving average window size
    train['Moving_Avg'] = train['y'].rolling(window=window, min_periods=1).mean()
    last_value = train['Moving_Avg'].iloc[-1]
    return np.full(periods, last_value).round()

# Load and preprocess inventory data
inventory = pd.read_excel('.\14_Demand_Data_2022_2024.xlsx')
inventory = inventory.melt(id_vars='Material No.', var_name='Date', value_name='Quantity')
inventory['Date'] = pd.to_datetime(inventory['Date'])
inventory.sort_values(by=['Material No.', 'Date'], inplace=True)
inventory.reset_index(drop=True, inplace=True)

# Initialize lists to store forecasts and error metrics
sarimax_forecasts = []
xgboost_forecasts = []
moving_avg_forecasts = []
error_metrics = {'Material No.': [], 'Model': [], 'MAE': [], 'RMSE': []}

# Determine the start month for forecasting
latest_date = inventory['Date'].max()
start_month = latest_date + pd.DateOffset(months=1)

# Loop through each unique material number
for part in inventory['Material No.'].unique():
    part_data = inventory[inventory['Material No.'] == part]
    
    # Prepare and clean the data
    part_data_prepared = prepare_data(part_data)
    
    # Skip parts with insufficient data
    if len(part_data_prepared) < 18 or part_data_prepared['y'].nunique() <= 1:
        print(f"Not enough data for Material No.: {part}")
        continue
    
    # Split data into training and test sets
    train = part_data_prepared.copy()
    test = part_data_prepared[-12:]
    test_values = test['y'].values

    # SARIMAX Forecast
    try:
        sarimax_forecast = forecast_sarimax(train[['ds', 'y']], periods=len(test))
        sarimax_forecasts.append({'Material No.': part, 'Forecast': list(sarimax_forecast)})
        sarimax_mae = mean_absolute_error(test_values, sarimax_forecast)
        sarimax_rmse = np.sqrt(mean_squared_error(test_values, sarimax_forecast))
        error_metrics['Material No.'].append(part)
        error_metrics['Model'].append('SARIMAX')
        error_metrics['MAE'].append(sarimax_mae)
        error_metrics['RMSE'].append(sarimax_rmse)
    except Exception as e:
        print(f"SARIMAX failed for Material No.: {part} with error: {e}")
    
    # XGBoost Forecast
    try:
        xgboost_forecast = forecast_xgboost(train[['ds', 'y']], periods=len(test))
        xgboost_forecasts.append({'Material No.': part, 'Forecast': list(xgboost_forecast)})
        xgboost_mae = mean_absolute_error(test_values, xgboost_forecast)
        xgboost_rmse = np.sqrt(mean_squared_error(test_values, xgboost_forecast))
        error_metrics['Material No.'].append(part)
        error_metrics['Model'].append('XGBoost')
        error_metrics['MAE'].append(xgboost_mae)
        error_metrics['RMSE'].append(xgboost_rmse)
    except Exception as e:
        print(f"XGBoost failed for Material No.: {part} with error: {e}")

    # Moving Average Forecast
    try:
        moving_avg_forecast = forecast_moving_average(train[['ds', 'y']], periods=len(test))
        moving_avg_forecasts.append({'Material No.': part, 'Forecast': list(moving_avg_forecast)})
        moving_avg_mae = mean_absolute_error(test_values, moving_avg_forecast)
        moving_avg_rmse = np.sqrt(mean_squared_error(test_values, moving_avg_forecast))
        error_metrics['Material No.'].append(part)
        error_metrics['Model'].append('Moving Average')
        error_metrics['MAE'].append(moving_avg_mae)
        error_metrics['RMSE'].append(moving_avg_rmse)
    except Exception as e:
        print(f"Moving Average failed for Material No.: {part} with error: {e}")

# Convert forecasts and error metrics to DataFrames and save them
sarimax_df = pd.DataFrame(sarimax_forecasts)
xgboost_df = pd.DataFrame(xgboost_forecasts)
moving_avg_df = pd.DataFrame(moving_avg_forecasts)
error_metrics_df = pd.DataFrame(error_metrics)

# Format forecast columns with Month-Year headers
month_year_columns = [(start_month + pd.DateOffset(months=i)).strftime('%b-%Y') for i in range(len(test))]
for df in [sarimax_df, xgboost_df, moving_avg_df]:
    if 'Forecast' in df.columns:
        df[month_year_columns] = pd.DataFrame(df['Forecast'].tolist(), index=df.index)
        df.drop(columns=['Forecast'], inplace=True)

# Save forecasts and error metrics to an Excel file
with pd.ExcelWriter('.\New_Forecast_All_Models.xlsx', engine='xlsxwriter') as writer:
    sarimax_df.to_excel(writer, sheet_name='SARIMAX', index=False)
    xgboost_df.to_excel(writer, sheet_name='XGBoost', index=False)
    moving_avg_df.to_excel(writer, sheet_name='Moving Average', index=False)
    error_metrics_df.to_excel(writer, sheet_name='Error Metrics', index=False)

print("Forecasts and error metrics saved successfully.")

# Calculate mean actual demand and determine the best model based on error metrics
mean_actual_demand = inventory.groupby('Material No.')['Quantity'].mean().reset_index()
mean_actual_demand.columns = ['Material No.', 'Mean Actual Demand']

# Prepare the MAE forecast_results DataFrame
mae_forecast_results = error_metrics_df.pivot(index='Material No.', columns='Model', values='MAE').reset_index()
mae_forecast_results.columns = ['Material No.', 'SARIMAX MAE', 'XGBoost MAE', 'Moving Avg MAE']

# Pivot the error_metrics_df to get RMSE for each model
rmse_forecast_results = error_metrics_df.pivot(index='Material No.', columns='Model', values='RMSE').reset_index()
rmse_forecast_results.columns = ['Material No.', 'SARIMAX RMSE', 'XGBoost RMSE', 'Moving Avg RMSE']

# Merge the mean actual demand with the MAE and RMSE forecast results
merged_results = pd.merge(mae_forecast_results, rmse_forecast_results, on='Material No.')
merged_results = pd.merge(merged_results, mean_actual_demand, on='Material No.')

# Calculate forecast accuracy based on MAE
for model in ['SARIMAX', 'XGBoost', 'Moving Avg']:
    merged_results[f'{model} MAE Accuracy'] = (1 - merged_results[f'{model} MAE'] / merged_results['Mean Actual Demand']) * 100
    merged_results[f'{model} RMSE Accuracy'] = (1 - merged_results[f'{model} RMSE'] / merged_results['Mean Actual Demand']) * 100

# Determine the minimum MAE and RMSE and the corresponding models
for metric in ['MAE', 'RMSE']:
    merged_results[f'Min {metric}'] = merged_results[[f'SARIMAX {metric}', f'XGBoost {metric}', f'Moving Avg {metric}']].min(axis=1)
    merged_results[f'Min {metric} Model'] = merged_results[[f'SARIMAX {metric}', f'XGBoost {metric}', f'Moving Avg {metric}']].idxmin(axis=1).str.replace(f' {metric}', '')

# Determine the best model based on a combination of MAE and RMSE
merged_results['Best Model'] = np.where(
    merged_results['Min MAE Model'] == merged_results['Min RMSE Model'],
    merged_results['Min MAE Model'],
    'Mixed'
)

# Select and rearrange the columns in the desired order
final_results = merged_results[['Material No.', 'Mean Actual Demand',
                                'SARIMAX MAE', 'XGBoost MAE', 'Moving Avg MAE',
                                'SARIMAX MAE Accuracy', 'XGBoost MAE Accuracy', 'Moving Avg MAE Accuracy', 'Min MAE', 'Min MAE Model',
                                'SARIMAX RMSE', 'XGBoost RMSE', 'Moving Avg RMSE',
                                'SARIMAX RMSE Accuracy', 'XGBoost RMSE Accuracy', 'Moving Avg RMSE Accuracy', 'Min RMSE', 'Min RMSE Model',
                                'Best Model']]

# Save the final merged results to an Excel file
final_results.to_excel('.\New_Forecast_Accuracy.xlsx', index=False)

print("Final results saved successfully.")