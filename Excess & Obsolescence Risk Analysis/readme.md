Excess and Obsolete (E&O) Inventory Management Project
Overview
This project aims to develop a predictive and data-driven approach to managing excess and obsolete (E&O) inventory for a leading chemical company. The solution involves identifying high-risk SKUs, implementing dynamic inventory adjustments, and creating actionable insights to reduce financial risks associated with excess and unsellable stock. The project uses machine learning models to predict E&O risk and visualizes the results in a Power BI dashboard for stakeholders.

Project Structure
Data Preparation: Data cleaning, feature engineering, and encoding were performed to prepare for modeling.
Modeling: Regression and classification models were built to identify key drivers of E&O risk and predict high-risk SKUs.
Evaluation: Model performance was assessed using metrics such as R-squared, MAE for regression, and F1 score, accuracy, and Jaccard index for classification.
Power BI Dashboard: A Power BI dashboard was developed to visualize insights, including key drivers, seasonality, SKU risk segmentation, and actionable recommendations.
Key Components
1. Data Preparation
Feature Engineering: Created time-based features, rolling averages, and lagged demand values to capture seasonality and demand trends.
Encoding: Categorical variables were encoded using Label Encoding, and mappings were maintained for interpretability.
Missing Value Handling: Median imputation was applied for missing values in numeric columns to ensure data consistency.
2. Machine Learning Models
Regression Models:
Gradient Boosting Regressor: Selected as the best model based on R-squared and MAE, effectively capturing non-linear relationships.
Random Forest Regressor: Also used for comparison, achieving similar results.
Classification Models:
Random Forest Classifier: Identified as the best model for classifying high-risk SKUs with high accuracy and F1 score.
Logistic Regression: Used as a baseline model to compare results.
3. Power BI Dashboard
The dashboard provides an interactive and visual representation of the insights from the model. Key sections include:

Overview: KPI cards highlighting total excess stock, high-risk SKU count, and average carrying cost.
Segmentation: Pie charts and bar charts showing risk distribution by lifecycle phase and subdivision.
Feature Influence: Visuals depicting the average impact of key features like Carrying Cost and Demand Fluctuation on E&O risk.
SKU Analysis: Detailed tables with filtering options to explore individual SKU metrics and risk levels.
Trend Analysis: Line charts displaying demand seasonality to guide inventory adjustments based on forecasted trends.
Usage
1. Model Training and Evaluation
The Python code provided in this repository handles data preprocessing, model training, and evaluation. The models are configured with hyperparameter tuning for optimal performance. To run the code:

Ensure all dependencies are installed (see Requirements).
Run the Jupyter notebook or Python scripts in the src directory for data preprocessing, model training, and evaluation.
Model results will be printed in the console and saved in a CSV file (dashboard_data.csv) for dashboard use.
2. Power BI Dashboard Setup
Open Power BI Desktop.
Load dashboard_data.csv into Power BI.
Follow the guide in the notebook to set up visuals, including KPI cards, bar charts, and line charts.
Requirements
Python 3.x
Libraries: pandas, numpy, scikit-learn, matplotlib
Power BI Desktop (for dashboard visualization)
Implementation Roadmap
Phase 1: Enhance Forecasting Accuracy – Improve forecast accuracy for SKUs with high variability.
Phase 2: Dynamic Safety Stock – Implement dynamic safety stock calculations.
Phase 3: Lifecycle Management – Prioritize SKUs in the Decline phase for inventory reduction.
Phase 4: Seasonal Adjustments – Adjust inventory levels based on seasonal trends.
Phase 5: Dashboard and Alerts – Integrate real-time dashboards and alerts for proactive management.
Key Insights & Recommendations
High Carrying Cost: SKUs with high carrying costs pose a higher risk of E&O; consider reducing stock levels or adjusting reorder points.
Demand Fluctuation: High variability in demand calls for dynamic forecasting models that adjust based on trends.
Proactive Lifecycle Management: Act on SKUs in the Decline phase to avoid unsellable stock.
Seasonality: Seasonal adjustments for products like Agrochemicals can reduce inventory waste.
Future Enhancements
Anomaly Detection: Integrate anomaly detection to identify sudden changes in SKU demand or lead times.
Automated Alerts: Add automated alerts for inventory planners to act on high-risk SKUs in real time.
Additional Data Integration: Integrate additional external factors, such as market trends or supplier reliability, to refine risk predictions.
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (feature/my-feature).
Commit changes.
Push to the branch.
Create a pull request.
