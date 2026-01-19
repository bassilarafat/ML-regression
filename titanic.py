from __future__ import absolute_import, division, print_function, unicode_literals

import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc

#load dataset link titanic dataset from kaggle
data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
evalData = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
print(data.head())    #pandas dataframe

data.describe()
# # extract the output column
# y_train = data.pop('survived')
# # extract the input features
# y_eval=evalData.pop('survived')
# print(data.head())
# print(data['age'].isnull().sum())

#graph data histogram of age
# data.Age.hist(bins=20)

#get the histogram about sex
# data.Sex.value_counts().plot(kind='barh')

#get the histogram about the percentage survived by Sex
pd.concat([data],axis=1).groupby('Sex').Survived.mean().plot(kind='barh').set_xlabel('Survived percentage')
plt.show()


