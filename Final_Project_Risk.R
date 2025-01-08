#Libraries
library(stringr)
library(ggplot2)
library(corrplot)
library(dplyr)
library(GGally)
library(car)
library(randomForest)
library(caret)
library(MASS)
library(klaR)
library(gbm)
library(stargazer)
library(e1071) 


set.seed(123)
data = read.csv('Automobile data.csv')


#########################EXPLORATORY DATA ANALYSIS#########################################
summary(data)

#####--- Missing Values--- ######
#MVs: There are some variables that have special characters (?), that really are missing values:
mvs = data.frame(column = character(0), mvs = integer(0))
column_names = colnames(data)

for (i in 1:length(column_names)) {
  column_name = column_names[i]
  number_mvs = sum(str_count(data[,i], fixed("?")))
  new_row = data.frame(column = column_name, mvs = number_mvs )
  mvs = rbind(mvs, new_row)
}


columns_mvs = mvs[mvs$mvs>0,1]
data[,columns_mvs] = lapply(data[,columns_mvs], function(x) ifelse(x == "?", NA, x)) #Replace the "?" for NA in all the columns identified with MVs

missing_rows <- data[!complete.cases(data), ]

#Let's identify the variables that are continuous and categorical (and make corrections accordingly)

sapply(data, class)
data$normalized.losses = as.numeric(data$normalized.losses)
data$bore = as.numeric(data$bore)
data$stroke = as.numeric(data$stroke)
data$horsepower = as.numeric(data$horsepower)
data$peak.rpm = as.numeric(data$peak.rpm)
data$price = as.numeric(data$price)


#First, let's try to deal with normalized.losses

#Correlation matrix: To identify the variables that are strongly correlated and make an informed data imputation

continuous_vars = c()

for (i in 1:ncol(data)) {
  # Check if the class of the column is either 'integer' or 'numeric'
  if (class(data[, i]) %in% c('integer', 'numeric')) {
    # Append the column name to the continuous_vars vector
    continuous_vars <- c(continuous_vars, colnames(data)[i])
  }
}

continuous_no_mvs = na.omit(data[,continuous_vars])
cor_matrix = cor(continuous_no_mvs)
par(mar = c(5, 4, 7, 2))
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
title(main = "Correlation Matrix", line = 3, cex.main = 2)

data_imputation = data[,c('normalized.losses','height')]

ggplot(data = data, mapping = aes(y = height, x = normalized.losses)) + 
  geom_point() + 
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE) +
  labs(title = 'Normalized Losses vs Height', x = 'Normalized Losses', y = 'Height')

# Build a linear regression model excluding missing values in the target variable (height)
lm_model <- lm(normalized.losses ~ height, data = data, na.action = na.exclude)

# Impute missing values of normalized losses using the regression model
data$normalized.losses[is.na(data$normalized.losses)] <- predict(lm_model, newdata = data[is.na(data$normalized.losses), ])



#num.of.doors: only 2 missing values
ggplot(data = data, mapping = aes(x=num.of.doors)) + geom_bar()
data$num.of.doors[is.na(data$num.of.doors)] = 'four'

#bore: only 4 missing values
ggplot(data = data, mapping = aes(x=bore, y = horsepower)) + geom_point() +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE)

lm_model <- lm(bore ~ horsepower, data = data, na.action = na.exclude)
data$bore[is.na(data$bore)] <- predict(lm_model, newdata = data[is.na(data$bore), ])

#stroke: only 4 missing values (since there is no strong correlation with other variables)
data$stroke[is.na(data$stroke)] <- median(data$stroke, na.rm = TRUE)

#Horsepower: only 2 missing values
ggplot(data = data, mapping = aes(x = city.mpg, y = horsepower)) + 
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = TRUE) +
  labs(title = "Regression with Squared Term",
       x = "City MPG",
       y = "Horsepower")

lm_model <- lm(horsepower ~ poly(city.mpg,2), data = data, na.action = na.exclude)
data$horsepower[is.na(data$horsepower)] <- predict(lm_model, newdata = data[is.na(data$horsepower), ])

#peak.rpm: only 2
data$peak.rpm[is.na(data$peak.rpm)] <- median(data$peak.rpm, na.rm = TRUE)

#price: 4 missing values
ggplot(data = data, mapping = aes(x = curb.weight, y = price)) + 
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x, se = TRUE) +
  labs(title = "Regression Curb Weight vs Price",
       x = "Curb Weight",
       y = "Price")

lm_model <- lm(price ~ curb.weight, data = data, na.action = na.exclude)
data$price[is.na(data$price)] <- predict(lm_model, newdata = data[is.na(data$price), ])


########## --- Multicollinearity Issues --- ##########
# Identify and convert all categorical columns to factors
data[sapply(data, function(x) is.character(x) || is.factor(x))] <- 
  lapply(data[sapply(data, function(x) is.character(x) || is.factor(x))], as.factor)

data_continuous <- data[sapply(data, is.numeric)]
#ggpairs(data_continuous)

#Let's Explore the categorical variables and check for unary or nearly unary columns
table(data$symboling)
table(data$make)
table(data$fuel.type)
table(data$aspiration)
table(data$num.of.doors)
table(data$body.style)
table(data$drive.wheels)
table(data$engine.location) #202 are located in front and 3 in rear 
table(data$engine.type)
table(data$num.of.cylinders)
table(data$fuel.system)


data = subset(data, select = -engine.location)

#Categorical Variables
categorical_vars <- sapply(data, function(x) is.factor(x) || is.character(x))
categorical_vars <- colnames(data)[categorical_vars]
summary(data$num.of.cylinders)
lm_model = lm(price ~ num.of.cylinders, data = data)
summary(lm_model)

lm_model = lm(price ~ ., data = data)
summary(lm_model)

#There are only 2 types of fuel types, diesel and gas. All the diesel vehicles have an idi fuel system, and 
#also all idi fuel systems have a diesel fuel type. The rest of the vehicles use gas as fuel type, therefore,
#the info of one variable is already represented in the other variable.
data = subset(data, select = -fuel.type)
lm_model = lm(price ~ num.of.cylinders + engine.type, data = data)
summary(lm_model)
data_1 = data.frame(data$num.of.cylinders, data$engine.type)


#Since there are multicollinearity issues with the variable cylinders and because that variable is not evenly splitted,
#a possible approach to model this variable is redefine the categories (if a car has 4 or less cylinders, 5 or more cylinders)
data$num.of.cylinders = ifelse(data$num.of.cylinders %in% c('two', 'three', 'four'), '4 or less', '5 or more')
table(data$num.of.cylinders)

#VIF to remove features
lm_model <- lm(symboling ~., data = data)
vif(lm_model)

lm_model <- lm(symboling ~.- compression.ratio - city.mpg - engine.size - curb.weight - wheel.base - price, data = data)
summary(lm_model)
vif_values = vif(lm_model)
vif_df <- data.frame(vif_values)
vif_df <- subset(vif_df, select = -GVIF)

# Create a stargazer table
stargazer(vif_df, type = 'html', summary = FALSE, rownames = TRUE)

data = subset(data, select = -c(compression.ratio, city.mpg, engine.size, curb.weight, wheel.base, price))
#Conclusion: 
#1. The variable fuel.type was dropped and the variable num.cylinders was redefined to deal with multicollinearity issues.
#2. The variable compression.ratio was removed due to a high VIF value, city.mpg is also deleted not only because of its VIF but because of its nearly 1 correlation with highway.mpg

#Let';s check again the correlation matrix
continuous_vars = c()

for (i in 1:ncol(data)) {
  # Check if the class of the column is either 'integer' or 'numeric'
  if (class(data[, i]) %in% c('integer', 'numeric')) {
    # Append the column name to the continuous_vars vector
    continuous_vars <- c(continuous_vars, colnames(data)[i])
  }
}

continuous_vars = data[,continuous_vars]
cor_matrix = cor(continuous_vars)
par(mar = c(5, 4, 7, 2))
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black")
title(main = "Correlation Matrix", line = 3, cex.main = 2)



###### --- TRANSFORMATION OF CONTINUOUS VARIABLES
#normalized.losses
summary(data$normalized.losses)
ggplot(data = data, aes(x = log(normalized.losses))) + geom_histogram(fill = "steelblue", color = "black", alpha = 0.7) +
  labs(
    title = "Log-Transformed Histogram of Normalized Losses",
    x = "Log of Normalized Losses",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )


#ggpairs(data_continuous)

#horsepower
ggplot(data = data, aes(x = log(horsepower))) + geom_histogram(fill = "steelblue", color = "black", alpha = 0.7) +
  labs(
    title = "Log-transformed Histogram of Horsepower",
    x = "Log of Horsepower",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )



#Conclusion: The variables horsepower and normalized losses were transformed using natural logarithmic

data$log_horsepower = log(data$horsepower)
data$log_normalized_losses = log(data$normalized.losses)


data = subset(data, select = -c(horsepower, normalized.losses))
colnames(data)

################Dependent Variable#######################
table(data$symboling) #Due to the reduced amount of data, it is much better to have a balanced dependent variable (risky or not risky)
ggplot(data = data, aes(x = symboling)) +
  geom_bar(fill = "steelblue", color = "black", width = 0.8) + # Add color and adjust bar width
  labs(
    title = "Distribution of Symboling",
    x = "Symboling",
    y = "Count"
  ) +
  theme_minimal(base_size = 14) + # Clean and modern theme
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16), # Center and bold title
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10)
  ) +
  scale_x_continuous(breaks = unique(data$symboling)) + # Ensure proper x-axis ticks
  geom_text(
    stat = "count", aes(label = ..count..), 
    vjust = -0.3, color = "black", size = 4
  ) # Add counts above bars
#Risky: 1,2,3
#Not Risky: -3, -2, -1, 0
data$symboling = ifelse(data$symboling %in% c(-3,-2,-1,0), 0, 1)
table(data$symboling)


ggplot(data = data, aes(x = factor(symboling))) +
  geom_bar(fill = "steelblue", color = "black") +
  geom_text(
    stat = "count", 
    aes(label = ..count..), 
    vjust = -0.3, # Adjust the vertical position of the labels
    color = "black", 
    size = 4
  ) +
  labs(
    title = "Count Plot of Symboling",
    x = "Symboling Levels",
    y = "Count"
  ) +
  scale_x_discrete(
    labels = c("Not Risky (0)", "Risky (1)") # Replace these labels with your desired names
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    axis.text.x = element_text(size = 10),
    axis.text.y = element_text(size = 10)
  )

##############FEATURE SELECTION BASED ON IMPORTANCE#################
data$symboling = as.factor(data$symboling)
rf_selection = randomForest(symboling~., data = data, importance = TRUE)
importance(rf_selection)

# Extract variable importance
importance_values <- importance(rf_selection)

importance_df <- as.data.frame(importance_values)
importance_df$Feature <- rownames(importance_df)

# Plot using ggplot2
ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Mean Decrease in Accuracy") +
  theme_minimal()

#Conclusion: Based on the results, the variables aspiration and num.of.cylinders are not that important for the model, therefore, they will be removed.
data = subset(data, select = -c(num.of.cylinders, aspiration))


##################PREDICTIVE MODELING################
#Clean the global environment
rm(data_continuous)
rm(data_imputation)
rm(importance_df)
rm(importance_values)
rm(lm_model)
rm(mvs)
rm(missing_rows)
rm(new_row)
rm(pca)
rm(rf_selection)
rm(continuous_no_mvs)
rm(cor_matrix)


#####---Random Forest---#######
# Define the tuning grid
tune_grid <- expand.grid(
  mtry = c(2, 3, 4, 5, 6, 7)          # Number of features tried at each split
)

# Train the model with cross-validation
control <- trainControl(method = "LOOCV")  # 10-fold cross-validation

rf_model <- train(
  symboling ~ .,                # Formula (replace with your formula)
  data = data,                  # Training dataset
  method = "rf",                # Random Forest method
  tuneGrid = tune_grid,         # Use the defined grid
  trControl = control,           # Cross-validation setup
  )

# View the best model
print(rf_model)
print(rf_model$bestTune)  # Best parameters


# Create a data frame with the results
rf_results <- rf_model$results

# Plot accuracy vs. mtry and ntree
ggplot(rf_results, aes(x = mtry, y = Accuracy)) +
  geom_line() + 
  geom_point() + 
  labs(title = "Hyperparameter Tuning: Accuracy vs. mtry",
       x = "mtry (number of features selected for doing the splits)",
       y = "Accuracy") +
  theme_minimal()

rf_best = randomForest(symboling~., data = data, importance = TRUE, mtry = 4)
importance_values <- importance(rf_best)

# Extract variable importance
importance_values <- importance(rf_best)

importance_df <- as.data.frame(importance_values)
importance_df$Feature <- rownames(importance_df)

# Plot using ggplot2
ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance", x = "Features", y = "Mean Decrease in Accuracy") +
  theme_minimal()

#####---Logistic Regression---#####
logit = glm(symboling ~ log_normalized_losses+num.of.doors+drive.wheels+length+fuel.system+bore, data = data, family = 'binomial')
summary(logit)

# Define cross-validation settings
control <- trainControl(method = "LOOCV")  # 10-fold cross-validation

# Fit GLM with cross-validation
cv_model <- train(symboling ~ log_normalized_losses+num.of.doors+drive.wheels+length+fuel.system+bore, 
                  data = data, 
                  method = "glm", 
                  family = "binomial", 
                  trControl = control)

# Print results
print(cv_model)

# Accuracy
accuracy <- cv_model$results$Accuracy
print(paste("Cross-validated Accuracy:", round(accuracy, 3)))

####---Boosting---#####
tune_grid <- expand.grid(
  n.trees = c(100, 300, 500),        # Number of trees
  interaction.depth = c(1, 3, 5),    # Depth of each tree
  shrinkage = c(0.01, 0.1),          # Learning rate
  n.minobsinnode = c(10, 20)
)

control <- trainControl(
  method = "LOOCV",           # Number of folds
  verboseIter = TRUE     # Print progress (optional)
)

gbm_model <- train(
  symboling ~ .,                  # Replace with your formula
  data = data,                    # Dataset
  method = "gbm",                 # Use GBM for boosting
  trControl = control,            # Cross-validation settings
  tuneGrid = tune_grid,           # Parameter grid
  verbose = FALSE                 # Suppress GBM output
)

# View results
print(gbm_model)
print(gbm_model$bestTune)

############## RESULTS : RANDOM FOREST ##################
rf_best = randomForest(symboling~., data = data, importance = TRUE, mtry = 4)

# Ensure symboling is a factor
data$symboling <- as.factor(data$symboling)

# Apply make.names() to clean up levels
levels(data$symboling) <- make.names(levels(data$symboling))

# Verify the levels
levels(data$symboling)

#Create the metrics to analyze prediction
custom_metrics <- function(data, lev = NULL, model = NULL) {
  # True Positive, False Positive, etc.
  confusion <- confusionMatrix(data$pred, data$obs, positive = lev[2])
  precision <- confusion$byClass["Precision"]
  recall <- confusion$byClass["Recall"]
  f1 <- confusion$byClass["F1"]
  
  c(Precision = precision, Recall = recall, F1 = f1)
}

control <- trainControl(
  method = "LOOCV",                 # Cross-validation                    
  summaryFunction = custom_metrics,  # Custom metrics
  classProbs = TRUE,             # Enable probabilities
  savePredictions = TRUE         # Save predictions for calculations
)


model <- train(
  symboling ~ .,                 # Formula
  data = data,                   # Dataset
  method = "rf",                 # Random Forest
  trControl = control,           # Cross-validation setup
  tuneGrid = expand.grid(mtry = 4)  # Set hyperparameter
)

print(model)

model$results

# Create the data frame with the results
results_df <- data.frame(
  Accuracy = c(0.941),
  Precision = c(0.9316239),
  Recall = c(0.9646018),
  F1 = c(0.9478261)
)

# Use stargazer to create the table
stargazer(results_df, 
          type = "html",  # Output as HTML
          summary = FALSE,  # Do not summarize, data is already summarized
          digits = 3,  # Number of decimal places
          column.sep.width = "1pt",  # Set the column separation width
          column.labels = c("Accuracy", "Precision", "Recall", "F1 Score"))  # Set custom column names


#Count plot of the cars per brand, filtered by their symboling classification
data$symboling <- factor(data$symboling, levels = c("X0",'X1'), labels = c("Not Risky", "Risky"))

ggplot(data = data, aes(x = make, fill = symboling)) + 
  geom_bar() +  # Create a bar plot
  labs(x = "Car Make", y = "Count") +  # Add axis labels
  theme_minimal() +  # Clean theme
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + # Rotate x-axis labels 90 degrees
  labs(title = 'Car Make filtering by Symboling') 


ggplot(data = data, aes(x = num.of.doors, fill = symboling)) + 
  geom_bar() +  # Create a bar plot
  labs(x = "Number of Doors", y = "Count") +  # Add axis labels
  theme_minimal() +  # Clean theme
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + # Rotate x-axis labels 90 degrees
  labs(title = "Car's Number of Doors Filtering by Symboling") 

ggplot(data = data, aes(x = symboling, y = length)) + geom_boxplot(fill = 'steelblue') + theme_minimal() + labs(title = 'Boxplot of Length for each Symboling')

ggplot(data = data, aes(x = symboling, y = log_normalized_losses)) + geom_boxplot(fill = 'steelblue') + theme_minimal() + 
  labs(title = 'Boxplot of Log (Normalized Losses) for each Symboling', y = "Log(Normalized Losses)", x = 'Symboling') +
  scale_x_discrete(labels = c("0" = "Not Risky", "1" = "Risky"))


############ PCA ##############

###################--PCA--####################
#Numerical Variables
data_continuous <- data[sapply(data, is.numeric)]

pca = prcomp(data_continuous, scale = TRUE)
pca

# Proportion of variance explained by each component
variance_explained <- pca$sdev^2 / sum(pca$sdev^2)
cumulative_variance <- cumsum(variance_explained)

#PCAplot
plot(cumulative_variance, type = "b", 
     main = "PC Analysis",
     xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained")

library(ggfortify)
data_continuous['symboling'] = data['symboling']

autoplot(pca, data = data_continuous, loadings = TRUE, loadings.label = TRUE, colour = 'symboling') + 
  scale_colour_manual(
    name = "Symboling",  # Title of the legend
    values = c("X0" = "purple", "X1" = "darkgreen"),  # Assign colors for the unique values
    labels = c("Not Risky", "Risky")  # Custom labels for the legend
  ) + labs(title = 'PCA - First 2 Components') + theme_minimal()


#Conclusion: Almost 70% of the dataset is explained by the first 2 components, since my data is not that large, probably having a small amount of variables would be helpful





