# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
import pymannkendall as mk
import xlsxwriter
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load inventory data from Excel file
inventory = pd.read_excel('.\14_Demand_Data_2022_2024.xlsx')

# Transform data to long format where each 'Material No.' has multiple 'Date'-'Quantity' pairs
inventory = inventory.melt(id_vars='Material No.', var_name='Date', value_name='Quantity')

# Convert 'Date' column to datetime format and sort data by 'Material No.' and 'Date' for time series analysis
inventory['Date'] = pd.to_datetime(inventory['Date'])
inventory.sort_values(by=['Material No.', 'Date'], inplace=True)

# Function to analyze each part's time series data for seasonality and trend
def analyze_part(part_data):
    """
    Analyzes the time series data for a single part to check for seasonality and trend.
    
    Parameters:
    part_data (DataFrame): A subset of the inventory DataFrame containing data for a single 'Material No.'.
    
    Returns:
    tuple: (seasonal_influenced, trend_influenced) where each is a boolean indicating 
           whether seasonality or trend influence is present.
    """
    # Set 'Date' as index and select 'Quantity' for decomposition
    part_data = part_data.set_index('Date')['Quantity']
    
    # Resample data to monthly frequency, filling missing months with zero
    part_data = part_data.resample('MS').sum().fillna(0)
    
    # Skip analysis if data is constant or less than 15 months (not enough data for reliable analysis)
    if part_data.nunique() == 1 or len(part_data) < 15:
        return False, False
    
    # Decompose time series into trend, seasonality, and residual components using STL
    result = STL(part_data, seasonal=13).fit()
    
    # Check if seasonal and trend components influence the data
    seasonal_influenced = is_seasonal(result.seasonal)
    trend_influenced = is_trending(part_data)
    return seasonal_influenced, trend_influenced

# Function to assess seasonality in the seasonal component
def is_seasonal(seasonal_component, significance_level=0.05):
    """
    Determines if the seasonal component is significant based on the Augmented Dickey-Fuller test.
    
    Parameters:
    seasonal_component (Series): The seasonal component of the decomposed time series.
    significance_level (float): Threshold for statistical significance.
    
    Returns:
    bool: True if seasonality is present, False otherwise.
    """
    # Check for constant seasonal component, if so return False
    if seasonal_component.nunique() == 1:
        return False
    
    # Perform Augmented Dickey-Fuller test to check for stationarity (non-constant seasonality)
    p_value = adfuller(seasonal_component)[1]
    return p_value < significance_level

# Function to detect trend using both ADF and Mann-Kendall trend tests
def is_trending(data, significance_level=0.05):
    """
    Determines if a trend is present in the data using the ADF and Mann-Kendall tests.
    
    Parameters:
    data (Series): The time series data for the part.
    significance_level (float): Threshold for statistical significance.
    
    Returns:
    bool: True if trend is present, False otherwise.
    """
    # Perform Augmented Dickey-Fuller test for trend stationarity
    adf_p_value = adfuller(data.dropna())[1]
    adf_trending = adf_p_value < significance_level
    
    # Perform Mann-Kendall test to detect monotonic trend
    mk_test = mk.original_test(data.dropna())
    mk_trending = mk_test.p < significance_level

    # Return True if either test detects a trend
    return adf_trending or mk_trending

# Dictionary to store seasonality and trend results for each 'Material No.'
analysis_results = {}

# Loop through each unique 'Material No.' and perform seasonality and trend analysis
for part in inventory['Material No.'].unique():
    part_data = inventory[inventory['Material No.'] == part]
    seasonal_influenced, trend_influenced = analyze_part(part_data)
    analysis_results[part] = {
        'Seasonal Influenced': seasonal_influenced,
        'Trend Influenced': trend_influenced
    }

# Convert results dictionary to a DataFrame for better visualization and export
results_df = pd.DataFrame.from_dict(analysis_results, orient='index')
results_df.index.name = 'Material No.'

# Display the results for verification
print(results_df)

# Save the results to an Excel file
results_df.to_excel('.\Stationarity_Trend_Test.xlsx', engine='xlsxwriter')
print("Results saved to Excel")
