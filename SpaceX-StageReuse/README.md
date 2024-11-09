# SpaceX - IBM Datascience Capstone Project

## Introduction


The objective of this project is to predict whether the Falcon 9 rocket's first stage will successfully land. SpaceX offers a cost-effective rocket launch option, charging $62 million per launch, whereas other companies charge upwards of $165 million. A significant factor in SpaceX’s cost savings is the reusability of the rocket's first stage. By accurately predicting whether the first stage will land and be reused, we can estimate the overall cost of a launch. This information can be crucial for other companies seeking to compete with SpaceX in the space launch industry.

## Business Problem Statement

SpaceX's competitive advantage in the space launch market is its ability to reuse the Falcon 9 rocket's first stage, allowing them to offer launches at a reduced price of $62 million, compared to competitors charging over $165 million. Predicting the likelihood of a successful first-stage landing can provide insight into launch costs.

As a data scientist employed by a competing startup, my role is to develop data-driven models that can accurately predict the probability of a successful first-stage landing. With these predictions, the company can make more informed and competitive bids against SpaceX for rocket launch contracts, ultimately gaining a strategic advantage in the market.

## Deliverables
Accurately Predict the Likelihood of the First Stage Rocket Landing Successfully: Leverage data science and machine learning techniques to build models that predict the probability of the Falcon 9's first-stage landing. This prediction is critical in estimating the total cost of a rocket launch.

Explore and Analyze the Data for Insights: Perform data exploration to uncover key patterns and insights that can provide a better understanding of the factors influencing the success of the rocket's first-stage landing, helping to optimize the model's predictions and provide actionable business intelligence.

## Methodology
The data was sourced from the SpaceX public API and publicly available resources like Wikipedia. The data wrangling process involved extracting launch outcome information, which served as the target variable for the Machine Learning models. Various SQL queries and data visualizations—including static plots, interactive maps, and a dashboard—were employed to uncover key insights. Predictive analysis was conducted using four Machine Learning models: Logistic Regression, Support Vector Machine (SVM), Decision Tree, and k-Nearest Neighbors (KNN).

## Results
The dataset, containing details such as flight number, launch date, payload mass, orbit type, launch site, and mission outcome, was thoroughly explored and visualized. Logistic Regression, SVM, and KNN models demonstrated similar performance in predicting launch outcomes. F1, Jaccard and Performance scores were similar for all models.
