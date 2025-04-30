# coded by Vahhab Balandomanesh (40215393)

# firt of all , we need to import all modules we want:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

# then , according to the given dataset, we put them to numpy arrays (reshape(-1,1) is because in the sk-learn we should define the dimension of features [features should be 2D])
x = np.array([-4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 20]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#and at the first step, we compute linear regression using the sk-learn module:
linear = LinearRegression()
# fitting the data to the model:
linear.fit(x, y)
# prediction part for linear regression:
linear_pred = linear.predict(x)
# linear model may not predict some datas to binary(because it fits the data to a line) so we should set the predictions to binary mode!
linear_pred_binary = (linear_pred >= 0.5).astype(int)

# at the second part, we make the logistic model:
logistic = LogisticRegression()
# again we fit the data to our model:
logistic.fit(x, y)
# and then the prediction part:
logistic_pred_binary = logistic.predict(x)

# then we compare the accuracy_score of our models
linear_accuracy = accuracy_score(y, linear_pred_binary)
logistic_accuracy = accuracy_score(y, logistic_pred_binary)

# printing the result:
print("Accuracy of Linear Regression:" , linear_accuracy)
print("Accuracy of Logistic Regression:" , logistic_accuracy)

# we choose 200 numbers from the range of our features (-4.5 , 20)(or for better plot shape (-6,22)) to draw the plot of linear and logistic models with the predictions of this range! 
x_range = np.linspace(-6, 22, 200).reshape(-1, 1)

# predictions of our range by each of the models:
linear_decision = linear.predict(x_range)
logistic_decision = logistic.predict_proba(x_range)[:, 1]
# setting the size of plot
plt.figure(figsize=(10, 6))

# put the data points on the plot
plt.scatter(x, y, color='black', label='Data points')

#drawing the predictions of each of the linear and the logistic model with the different colors
plt.plot(x_range, linear_decision, color='black', label='Linear Regression')
plt.plot(x_range, logistic_decision, color='orange', label='Logistic Regression')

# drawing the treshold line with mathplotlib 
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold = 0.5')

# at the end , some settings of the plot!
plt.title(" Linear vs Logistic Regression")
plt.xlabel("x")
plt.ylabel("Prediction Probability")
plt.legend()
plt.grid(True)
plt.show()