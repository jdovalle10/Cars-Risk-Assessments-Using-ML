# Cars-Risk-Assessments-Using-ML

## Overview
This project aims to develop a machine-learning model to predict car risk assessments, known as symboling, which is widely used in the automotive and insurance industries. Accurate predictions of symbolling enable insurers to tailor premiums correctly, and consumers to make more informed purchasing decisions. Furthermore, predicting symboling can assist fleet managers in selecting cost-effective vehicles, financial institutions in evaluating loan and lease terms, and manufacturers in improving the safety features of their cars. By leveraging machine-learning techniques, the accuracy of symboling predictions can be enhanced, allowing for more accurate risk assessments and better management of potential financial and operational impacts across various sectors. 

## Data Description
The automobile dataset consists of 205 observations and 26 features, that contain information about risk assessment and insurance, vehicle dimensions, performance and power metrics, efficiency and fuel consumption, and car specifications.

## Methodology
### Cleaning:
The dataset had missing values in 7 features, to deal with them different imputation techniques were applied. The removal of the observations was not an option, considering that the dataset had a small number of them. The following are just some examples of the imputation process for some variables:
  1. Normalized Losses: As it is semantically related to the car’s risk assessment, having an accurate normalized loss value is essential for predicting the symbolling. In this case, the feature had 41 missing values, which represented 20% of the total observations (there was no pattern identified in the missingness of data). To impute the missing values, a correlation matrix was calculated, and it was possible to determine that the feature “height” has a negative correlation of 0.42; therefore, a linear regression between these 2 features was performed to impute the missing values.
