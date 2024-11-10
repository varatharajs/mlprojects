# Demand Forecasting and Inventory Optimization for DataCo Global

## Overview
Navigating the ever-evolving landscape of customer demand is a critical challenge for supply chain professionals. This project presents a scalable, data-driven approach to demand forecasting using Python, aimed at optimizing inventory management for DataCo Global. By leveraging advanced time series modeling and incorporating essential inventory concepts, this project provides insights to support data-driven decisions in inventory control.

## Objectives
The project aimed to answer two key business questions:
1. What is the projected demand for the top-selling product over the next 24 months?
2. What is the optimal inventory level for this product to meet demand while minimizing costs?

## Project Outline
This project was conducted through the following key steps:
- **Exploratory Data Analysis**: Uncovered underlying patterns in the dataset to inform modeling.
- **Data Cleaning and Preparation**: Processed and structured the data for modeling.
- **Demand Forecasting with Prophet**: Developed a time series model to forecast demand for the top-selling product.
- **Model Evaluation**: Assessed model performance to confirm accuracy and reliability.
- **Inventory Policy Optimization**: Used the forecasting model to determine reorder points, safety stock, and Economic Order Quantity (EOQ) for optimal inventory management.

## Key Insights
- **Stable Demand Projection**: Demand for the top-selling product is projected to remain stable over the next two years, with observed seasonal dips each third quarter.
- **Inventory Policy Recommendations**:
  - **Reorder Point**: 3,722 units
  - **Economic Order Quantity (EOQ)**: 35 units
  - **Safety Stock**: 2,252 units

This policy suggests reordering at 3,722 units with an order size of 35 units to ensure adequate inventory levels. By implementing these recommendations, DataCo Global can better align inventory with demand, ensuring product availability while minimizing holding costs.

## Methodology
### Data Collection
- Dataset provided by DataCo Global, containing historical sales, lead time, and other inventory-relevant data.

### Tools and Libraries
- **Python**: Primary language for data analysis and model development.
- **Libraries**: `pandas`, `numpy`, `Prophet`, `scikit-learn`, `matplotlib`, `seaborn`

### Feature Engineering
- **Lead Time Analysis**: Differentiated upstream and downstream lead times to ensure accurate calculations for reorder points.
- **Seasonality and Holiday Effects**: Incorporated holidays and seasonal effects to refine demand forecasting accuracy.

## Model and Techniques
- **Prophet for Time Series Modeling**: Built a robust demand forecasting model using Prophet, capturing seasonality, holiday effects, and long-term trends.
- **Random Forest for Feature Importance**: Used Random Forest Regressor to identify the top predictors of demand, confirming that lead time and geographic factors were significant predictors.

## Conclusion
This project successfully demonstrated a data-driven approach to demand forecasting and inventory optimization. By developing a scalable model, DataCo Global can enhance its inventory management practices, reduce carrying costs, and adapt more effectively to fluctuations in demand.

## Future Enhancements
- **Automated Model Retraining**: Implement automated retraining to continuously improve forecast accuracy as new data becomes available.
- **Integration with Real-Time Data**: Link the forecasting model with real-time inventory and sales data to improve responsiveness to market changes.
- **Enhanced Feature Engineering**: Further exploration of potential predictors to improve model performance and gain deeper insights into demand drivers.

## Getting Started
### Prerequisites
- Python 3.7 or above
- Libraries: `pandas`, `numpy`, `Prophet`, `scikit-learn`, `matplotlib`, `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone [<repository_url>](https://github.com/varatharajs/mlprojects/tree/168bc43285247f98191477f11edf63be193b9664/Supply%20Chain%20Demand%20Forecasting%20-%20DataCo%20Global)
