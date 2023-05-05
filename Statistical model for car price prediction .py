#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv(r"C:\Users\16025\Downloads\CarDekho\car data.csv")
data.head(22)


# In[3]:


Age = 2019 - data.Year
data.insert(0, "Age", Age)
data.drop('Year', axis = 1, inplace = True)


# In[4]:


plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True)
plt.title('Heat Map', size=20)
plt.yticks(rotation = 0)
plt.show()


# In[5]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plt.bar(x='Age', height='Selling_Price', color='blue', data=data)
plt.title('Relation of Year with Selling Price', size=20)
plt.xlabel('Year', size=15)
plt.ylabel('Price', size=15)
plt.show()


# In[6]:


data["Fuel_Type"].replace({'Petrol':2, 'Diesel':3, 'CNG':4},inplace = True)
data["Seller_Type"].replace({'Dealer':2, 'Individual':3}, inplace = True)
data["Transmission"].replace({'Manual':2, 'Automatic':3}, inplace = True)
data.drop("Car_Name", axis=1, inplace = True)


# In[7]:


# Split the data into training and testing sets
x = pd.DataFrame(data, columns = ['Age','Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
y = data['Selling_Price'].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

# Lasso Regression Model without bias
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, Y_train)
y_lasso_train_pred = lasso.predict(X_train)
y_lasso_test_pred = lasso.predict(X_test)

# Lasso Regression Model with bias
lasso_cv = LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5, random_state=0)
lasso_cv.fit(X_train, Y_train)
y_lasso_cv_train_pred = lasso_cv.predict(X_train)
y_lasso_cv_test_pred = lasso_cv.predict(X_test)
print("Selected alpha value in LassoCV with bias model: ", lasso_cv.alpha_)
print("Bias value in Lasso with bias model: ", lasso_cv.intercept_)


# Evaluation Metrics for Lasso Regression Model without bias
print("Lasso Regression Model without bias:")
print("Mean Squared Error (MSE) train: {:.4f}".format(metrics.mean_squared_error(Y_train, y_lasso_train_pred)*1000))
print("Mean Squared Error (MSE) test: {:.4f}".format(metrics.mean_squared_error(Y_test, y_lasso_test_pred)*1000))
print("Mean Absolute Error (MAE) train: {:.4f}".format(metrics.mean_absolute_error(Y_train, y_lasso_train_pred)*1000))
print("Mean Absolute Error (MAE) test: {:.4f}".format(metrics.mean_absolute_error(Y_test, y_lasso_test_pred)*1000))
print("R-squared train: {:.4f}".format(metrics.r2_score(Y_train, y_lasso_train_pred)*100))
print("R-squared test: {:.4f}".format(metrics.r2_score(Y_test, y_lasso_test_pred)*100))

# Evaluation Metrics for Lasso Regression Model with bias
print("\nLasso Regression Model with bias:")
print("Mean Squared Error (MSE) train: {:.4f}".format(metrics.mean_squared_error(Y_train, y_lasso_cv_train_pred)*1000))
print("Mean Squared Error (MSE) test: {:.4f}".format(metrics.mean_squared_error(Y_test, y_lasso_cv_test_pred)*1000))
print("Mean Absolute Error (MAE) train: {:.4f}".format(metrics.mean_absolute_error(Y_train, y_lasso_cv_train_pred)*1000))
print("Mean Absolute Error (MAE) test: {:.4f}".format(metrics.mean_absolute_error(Y_test, y_lasso_cv_test_pred)*1000))
print("R-squared train: {:.4f}".format(metrics.r2_score(Y_train, y_lasso_cv_train_pred)*100))
print("R-squared test: {:.4f}".format(metrics.r2_score(Y_test, y_lasso_cv_test_pred)*100))


# In[8]:


import matplotlib.pyplot as plt

# Data for the bar charts
lasso = [86.8268, 91.4154]
lasso_cv = [86.8268, 91.4154]

# Create the first bar chart
plt.bar(x=['Train', 'Test'], height=lasso, color='blue')
plt.title('R-Squared for LASSO Regression Model (without bias)')
plt.xlabel('Data Split')
plt.ylabel('R-Squared')
plt.ylim([80, 100])
plt.show()

# Create the second bar chart
plt.bar(x=['Train', 'Test'], height=lasso_cv, color='red')
plt.title('R-Squared for Lasso Regression Model with bias')
plt.xlabel('Data Split')
plt.ylabel('R-Squared')
plt.ylim([80, 100])
plt.show()


# In[9]:


# Importing XGBoost
from xgboost import XGBRegressor

# Base Model of XGBoost Regressor
xgb_base = XGBRegressor()
xgb_base.fit(X_train, Y_train)
y_xgb_base_train_pred = xgb_base.predict(X_train)
y_xgb_base_test_pred = xgb_base.predict(X_test)

# Evaluation Metrics for Base Model of XGBoost Regressor
print("Base Model of XGBoost Regressor:")

print("Mean Squared Error (MSE) test: {:.4f}".format(metrics.mean_squared_error(Y_test, y_xgb_base_test_pred)*1000))

print("Mean Absolute Error (MAE) test: {:.4f}".format(metrics.mean_absolute_error(Y_test, y_xgb_base_test_pred)*1000))

print("R-squared test: {:.4f}".format(metrics.r2_score(Y_test, y_xgb_base_test_pred)*100))


# In[11]:


import matplotlib.pyplot as plt

# Data for the bar chart
r_squared = [98.9996, 96.3142]

# Create the bar chart
plt.bar(x=['Train', 'Test'], height=r_squared, color='green')
plt.title('R-Squared for XGBoost Model')
plt.xlabel('Data Split')
plt.ylabel('R-Squared')
plt.ylim([90, 100])
plt.show()


# In[12]:


import matplotlib.pyplot as plt

# Scatter plot for LASSO Regression Model without bias
plt.scatter(Y_test, y_lasso_test_pred)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('LASSO Regression Model without bias')
plt.show()

# Scatter plot for Lasso Regression Model with bias
plt.scatter(Y_test, y_lasso_cv_test_pred)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Lasso Regression Model with bias')
plt.show()


# In[13]:


# Importing GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Base Model of Gradient Boosting Regressor
gb_base = GradientBoostingRegressor()
gb_base.fit(X_train, Y_train)
y_gb_base_train_pred = gb_base.predict(X_train)
y_gb_base_test_pred = gb_base.predict(X_test)

# Evaluation Metrics for Base Model of Gradient Boosting Regressor
print("Base Model of Gradient Boosting Regressor:")
print("Mean Squared Error (MSE) train: {:.4f}".format(metrics.mean_squared_error(Y_train, y_gb_base_train_pred)*1000))
print("Mean Squared Error (MSE) test: {:.4f}".format(metrics.mean_squared_error(Y_test, y_gb_base_test_pred)*1000))
print("Mean Absolute Error (MAE) train: {:.4f}".format(metrics.mean_absolute_error(Y_train, y_gb_base_train_pred)*1000))
print("Mean Absolute Error (MAE) test: {:.4f}".format(metrics.mean_absolute_error(Y_test, y_gb_base_test_pred)*1000))
print("R-squared train: {:.4f}".format(metrics.r2_score(Y_train, y_gb_base_train_pred)*100))
print("R-squared test: {:.4f}".format(metrics.r2_score(Y_test, y_gb_base_test_pred)*100))


# In[14]:


import matplotlib.pyplot as plt

# Scatter plot for Base Model of Gradient Boosting Regressor
plt.scatter(Y_test, y_gb_base_test_pred)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title('Base Model of Gradient Boosting Regressor')
plt.show()


# In[18]:


import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder

# Load the training data
df = pd.read_csv(r"C:\Users\16025\Downloads\CarDekho\car data.csv")

# Split data into features and target variables
features = df[["Year", "Kms_Driven", "Owner"]]
target = df["Selling_Price"]

# One-hot encode the car name column
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_car_names = encoder.fit_transform(df[['Car_Name']])

# Concatenate the encoded car names with the other features
features = pd.concat([features, pd.DataFrame(encoded_car_names.toarray())], axis=1)

# Fit the XGBoost Regressor model to the training data
regressor = XGBRegressor()
regressor.fit(features, target)

# Make the prediction for a single car
new_df = pd.DataFrame({
    "Year": [2013],
    "Kms_Driven": [55.0],
    "Owner": [1],
    "Car_Name": ["swift"]
})

# One-hot encode the car name column for the new data
encoded_new_car_names = encoder.transform(new_df[['Car_Name']])

# Concatenate the encoded car names with the other features for the new data
new_features = pd.concat([new_df[["Year", "Kms_Driven", "Owner"]], pd.DataFrame(encoded_new_car_names.toarray())], axis=1)

price_prediction = regressor.predict(new_features)
print("Predicted price of Swift (2013):", price_prediction[0])


# In[ ]:


"#Scatter plot for Gradient Boosting

plt.scatter(Y_test, y_gb_test_pred) plt.xlabel('Real Values') plt.ylabel('Predicted Values') plt.title('Gradient Boosting Model') plt.show()
#Scatter plot for XGBoost

plt.scatter(Y_test, y_xgb_test_pred) plt.xlabel('Real Values') plt.ylabel('Predicted Values') plt.title('XGBoost Model') plt.show()
#Scatter plot for Random Forest base model

plt.scatter(Y_test, y_rf_test_pred) plt.xlabel('Real Values') plt.ylabel('Predicted Values') plt.title('Random Forest base model') plt.show()

#Scatter plot for Random Forest hyperparameter tuned with random search

plt.scatter(Y_test, y_rf_random_test_pred) plt.xlabel('Real Values') plt.ylabel('Predicted Values') plt.title('Random Forest hyperparameter tuned with random search') plt.show()"

 "#Random Forest base model

from sklearn.ensemble import RandomForestRegressor rf = RandomForestRegressor(random_state=0) rf.fit(X_train, Y_train) y_rf_train_pred = rf.predict(X_train) y_rf_test_pred = rf.predict(X_test)

#Evaluation Metrics for Random Forest base model

print("\nRandom Forest base model:") print("Mean Squared Error (MSE) train: {:.4f}".format(metrics.mean_squared_error(Y_train, y_rf_train_pred)*1000)) print("Mean Squared Error (MSE) test: {:.4f}".format(metrics.mean_squared_error(Y_test, y_rf_test_pred)*1000)) print("Mean Absolute Error (MAE) train: {:.4f}".format(metrics.mean_absolute_error(Y_train, y_rf_train_pred)*1000)) print("Mean Absolute Error (MAE) test: {:.4f}".format(metrics.mean_absolute_error(Y_test, y_rf_test_pred)*1000)) print("R-squared train: {:.4f}".format(metrics.r2_score(Y_train, y_rf_train_pred)*100)) print("R-squared test: {:.4f}".format(metrics.r2_score(Y_test, y_rf_test_pred)*100))

#Random Forest hyperparameter tuned with random search

from sklearn.model_selection import RandomizedSearchCV from scipy.stats import randint as sp_randint param_dist = {"n_estimators": sp_randint(10, 100), "max_depth": sp_randint(5, 15), "min_samples_split": sp_randint(2, 10), "min_samples_leaf": sp_randint(1, 10), "max_features": sp_randint(1, 5)}

n_iter_search = 20 rf_random = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter_search, cv=5, random_state=0) rf_random.fit(X_train, Y_train) y_rf_random_train_pred = rf_random.predict(X_train) y_rf_random_test_pred = rf_random.predict(X_test)

#Evaluation Metrics for Random Forest hyperparameter tuned with random search

print("\nRandom Forest hyperparameter tuned with random search:") print("Mean Squared Error (MSE) train: {:.4f}".format(metrics.mean_squared_error(Y_train, y_rf_random_train_pred)*1000)) print("Mean Squared Error (MSE) test: {:.4f}".format(metrics.mean_squared_error(Y_test, y_rf_random_test_pred)*1000)) print("Mean Absolute Error (MAE) train: {:.4f}".format(metrics.mean_absolute_error(Y_train, y_rf_random_train_pred)*1000)) print("Mean Absolute Error (MAE) test: {:.4f}".format(metrics.mean_absolute_error(Y_test, y_rf_random_test_pred)*1000)) print("R-squared train: {:.4f}".format(metrics.r2_score(Y_train, y_rf_random_train_pred)*100)) print("R-squared test: {:.4f}".format(metrics.r2_score(Y_test, y_rf_random_test_pred)*100))"
