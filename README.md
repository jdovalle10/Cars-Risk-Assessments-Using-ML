# Cars-Risk-Assessments-Using-ML

## Overview
This project aims to develop a machine-learning model to predict car risk assessments, known as symboling, which is widely used in the automotive and insurance industries. Accurate predictions of symbolling enable insurers to tailor premiums correctly, and consumers to make more informed purchasing decisions. Furthermore, predicting symboling can assist fleet managers in selecting cost-effective vehicles, financial institutions in evaluating loan and lease terms, and manufacturers in improving the safety features of their cars. By leveraging machine-learning techniques, the accuracy of symboling predictions can be enhanced, allowing for more accurate risk assessments and better management of potential financial and operational impacts across various sectors. 

## Data Description
The automobile dataset consists of 205 observations and 26 features, that contain information about risk assessment and insurance, vehicle dimensions, performance and power metrics, efficiency and fuel consumption, and car specifications.

## Methodology
### Cleaning:
The dataset had missing values in 7 features, to deal with them different imputation techniques were applied. The removal of the observations was not an option, considering that the dataset had a small number of them. The following are just some examples of the imputation process for some variables:
  1. Normalized Losses: As it is semantically related to the car’s risk assessment, having an accurate normalized loss value is essential for predicting the symbolling. In this case, the feature had 41 missing values, which represented 20% of the total observations (there was no pattern identified in the missingness of data). To impute the missing values, a correlation matrix was calculated, and it was possible to determine that the feature “height” has a negative correlation of 0.42; therefore, a linear regression between these 2 features was performed to impute the missing values.
  2. Stroke; Number of Doors; Peak RPM: These features had less than 4 missing values each. The median of each feature was used to fill the missing data.
### Muticollinearity Issues: 
Correlation matrices and Variance Inflation Factors were considered to identify the highly correlated variables; the variables that were considered highly correlated were deleted from the dataset, as most of their info is already captured in other variables.
### Feature Transformation:
The variables “Horsepower” and “Normalized Losses” were right skewed, which means that some observations have large values for these features, which may affect the prediction. To reduce the impact of those observations a logarithmic transformation was applied over those features.
### Dependent Variable Transformation:
Symboling categorizes cars based on their safety, reliability, and overall risk, with higher values indicating greater risk relative to their price. The range of the variable is from -3 to 3. Due to the reduced amount of data, is much better to have a balanced dependent variable (when the dependent variable is imbalanced, predictive models can become biased toward the majority class, leading to poor performance for the minority class). In line with that, a redefinition of the target variable was made, in this case, the dependent variable will only have two categories, “Risky” (1, 2, 3) and “Not Risky” (-3, -2, -1, 0).
### Feature Selection
The features that will be used in the predictive modeling were first tested through a standard Random Forest (without hyperparameter tunning), to determine the importance of each variable. The features “aspiration” and “num.of.cylinders”, when shuffled have a really small impact on the model’s accuracy (almost a 3% decrease); and since we have a small number of observations, these features were removed.
### Model Selection
For the model selection process, 3 models were pre-selected for prediction (Random Forest, Boosting and Logistic Regression). For the tree-based methods, a grid search approach was used to tune some relevant hyperparameters; for logistic regression there was no tunning. For each model Leave-One-Out Cross-Validation (LOOCV) was applied to estimate the test accuracy, and, in the case of the tree-based methods, determine the best combination of hyperparameters. LOOCV is particularly useful for small datasets, as it allows the model to be trained on almost all the data. The following were the results (estimated test accuracy):
1. Logistic Regression: 0.859
2. Random Forest: 0.941
3. Boosting: 0.922
Based on the previous results, Random Forest was the selected predictive model.
## Results
The following are the complete results of the Random Forest with the best hyperparameters identified in the previous section (estimated using LOOCV):
| Accuracy | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| 0.941    | 0.932     | 0.965  | 0.948    |

In general, the predictive model, had a good performance in all metrics; for precision the model got 0.932 which means that 93.2% of the cars that the model predicted as “Risky” were indeed correct; for recall, the model got 0.965, which means that 96.5% of all the cars that were in fact Risky were correctly identified by the model. In the context of our model, the F1 Score of 0.948 means that the model strikes a good balance between precision and recall. Finally, in terms of accuracy, the model got 0.941, which means that 94.1% of the predictions made by the model were correct.

## Conclusions
- Model Performance: Random Forest delivered the highest accuracy among tested predictive modeling techniques.
- Insurance Insights: Insurers can use the model to create tiered premium structures, charging higher rates for riskier vehicles and offering discounts for safer ones.
- Market Applications: Manufacturers can enhance product design, safety features, and marketing strategies, while customers gain insights to make informed purchasing decisions based on safety and cost considerations.

