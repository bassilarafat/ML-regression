import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('student-mat.csv',sep=';')

# print(data.head())

# Select relevant columns
data = data[['G1','G2','G3','studytime','failures','absences']]
print(data.head())
# Define features and target variable
predict = 'G3'

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Split the data into training and testing sets
x_train,y_train, x_test,y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.2)

# Train the model using Multiple Linear Regression
regression = linear_model.LinearRegression()
regression.fit(x_train, y_train)
pred = regression.predict(x_test)
print('pred', pred)
print('Mean squared error: %.2f' % mean_squared_error(y_test, pred))
print('R2 score: %.2f' % r2_score(y_test, pred))
for i in range(len(pred)):
    print(f'Predicted: {pred[i]}, Actual: {y_test[i]}')
plt.scatter(y_test, pred, color='black')
plt.plot([0, 20], [0, 20], color='blue', linewidth=3)
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.xticks(())
plt.yticks(())
plt.show()
