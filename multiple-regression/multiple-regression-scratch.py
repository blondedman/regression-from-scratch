import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error 
from sklearn.model_selection import train_test_split

# loading dataset
sales = pd.read_csv("multiple-regression/sales.csv", dtype = np.float64)

print(sales.head())

sales['TV'] = np.log1p(sales['TV'])
sales['radio'] = np.log1p(sales['radio'])
sales['newspaper'] = np.log1p(sales['newspaper'])

print(sales.head())
print(sales.shape)

# visualizing data
"""
plt.figure(figsize = (15,5))
plt.subplot(3,3,1)
sns.regplot(x = 'TV', y ='sales', data = sales, marker = 'x', color = 'lightblue')
plt.subplot(3,3,2)
sns.regplot(x = 'radio', y ='sales', data = sales, marker = 'x', color = 'lightblue')
plt.subplot(3,3,3)
sns.regplot(x = 'newspaper', y ='sales', data = sales, marker = 'x', color = 'lightblue')
"""
# plt.show()


# multiple regression
class multipleregression:
  
  def __init__(self, rate = 0.001, w = 3, b = 0.01*np.random.rand()):
    self.rate = rate
    self.w = np.random.rand(w)
    self.b = b
    
  def MLR(self, features):
    return (features@self.w) + self.b

  def loss(self, groundtruth, predictions):
    return np.sqrt(np.mean(np.square((groundtruth - predictions))))
  
  def gradientdescent(self, features, groundtruth, predictions):
    error = groundtruth - predictions
    dW = []
    for column in features.columns:
      dW.append(-2 * np.mean((error * features[column])))
    db = -2 * np.mean(error)
    return dW, db
  
  def optimizemodelparameters(self, features, groundtruth, predictions):
    dW, db = self.gradientdescent(features, groundtruth, predictions)
    self.w[0] += self.rate * (-dW[0])
    self.w[1] += self.rate * (-dW[1])
    self.b += self.rate * (-db)
    
  def fit(self, features, labels, epochs = 10, out = False):
    ytrue = labels
    history = {'epoch': [], 'loss': []}
    for epoch in range(epochs):
      yhat = self.MLR(features)
      loss = self.loss(ytrue, yhat)
      self.optimizemodelparameters(features, ytrue, yhat)
      if out:
        print('epoch:', epoch, 'loss:', loss)
      history['epoch'].append(epoch)
      history['loss'].append(loss)
    return history
  
  def modelcoef(self):
    return self.w, self.b
  
  def predict(self, features):
    return self.MLR(features)
  
  def evaluate(self, features, labels):
    ytest = labels
    yhat = self.predict(features)
    loss = self.loss(ytest, yhat)
    return loss
    
  
X = sales[['TV','radio','newspaper']]
y = sales['sales']

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

model = multipleregression()

history = model.fit(X_train, y_train, epochs = 10000, out=False)

# function to plot learning curve
def learningcurve(model, history):
  coef, bias = model.modelcoef()
  plt.figure(figsize = (8,5))
  plt.plot(history['loss']);
  plt.title(f'learning curve # learned weights:{coef} and bias:{bias :.2f}')
  plt.xlabel('epochs')
  plt.ylabel('mean squared error')
  plt.show()

learningcurve(model, history)

# predicting test values
predictions = model.predict(X_test)
print('MAE ', mean_absolute_error(y_test, predictions))
print('RMSE', root_mean_squared_error(y_test, predictions))

# same as RMSE from sklearn
# idk why does the code exist
loss = model.evaluate(X_test, y_test)
print('loss', loss)