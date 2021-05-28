import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# See first five records of dataset !
print(diabetic.head())

# Extract Age & Outcome Column from dataset !
diabetic = diabetic[['Age','Outcome']]

# Split the model into training & testing

x = diabetic[['Age']]
y = diabetic[['Outcome']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

# K-Means Implementation !
from sklearn.cluster import KMeans
km = KMeans()
print(km.fit(x_train,y_train))
y_pred = km.predict(x_test)

# Actual Values !
print(y_test.head())

# Predicted Values !
print(y_pred[0:5])

# Finding the residual !

from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,y_pred))