import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_x ,diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_x = diabetes_x[:, np.newaxis, 2]

diabetes_x_train = diabetes_x[:-20]  # Use all but the last 20 data points for training
diabetes_x_test = diabetes_x[-20:]   # Use the last 20 data points for testing

diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Train the model using linear regression
regression = linear_model.LinearRegression()
regression.fit(diabetes_x_train, diabetes_y_train)

pred = regression.predict(diabetes_x_test)

print('pred', pred)
print('Mean squared error: %.2f' % mean_squared_error(diabetes_y_test, pred))
print('R2 score: %.2f' % r2_score(diabetes_y_test, pred))
plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
plt.plot(diabetes_x_test, diabetes_y_test, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()
