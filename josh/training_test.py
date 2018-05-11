from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset and show basic data
dataset = pd.read_csv('data/california_housing_train.csv')
print(dataset.describe())
dataset.hist('total_rooms')

#plt.show()

#create scatterplot of basic data
x = dataset[['total_rooms']].values
y = dataset[['median_house_value']].values
plt.scatter(x, y)

plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

#set up linear regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

#create the linear regression fit
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Total room vs median house value (training set)')
plt.xlabel('total_rooms')
plt.ylabel('median_house_value')
plt.show()

#apply the fit to the test set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Total room vs median house value (test set)')
plt.xlabel('total_rooms')
plt.ylabel('median_house_value')
plt.show()




