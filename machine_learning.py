import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data
dataset = pd.read_csv('california_housing_train.csv')

#IMPORTANT: must use print to show basic dataset
print(dataset.describe())

#show histogram
dataset.hist()

#plot with matplotlib
X = dataset[['total_rooms']].values
Y = dataset[['median_house_value']].values

#plt.scatter(X,Y)

#IMPORTANT: required to show plots
plt.show()

