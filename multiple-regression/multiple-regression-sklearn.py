import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# loading dataset
sales = pd.read_csv("multiple-regression/sales.csv")

print(sales.head())
print(sales.shape)


# outlier analysis
fig, axs = plt.subplots(3, figsize = (10,5))
plt1 = sns.boxplot(sales['TV'], ax = axs[0], orient='h')
plt2 = sns.boxplot(sales['newspaper'], ax = axs[1], orient='h')
plt3 = sns.boxplot(sales['radio'], ax = axs[2], orient='h')
plt.tight_layout()
# plt.show()

# correlation graph
sns.pairplot(sales, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=4, aspect=1, kind='scatter')
# plt.show()

x = sales[['TV', 'radio', 'newspaper']]
y = sales['sales']

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.3, random_state=42)

mlr = LinearRegression()  
mlr.fit(x_train, y_train) 

# model coefficients
print(mlr.intercept_)

# feature names with coefficients
print(list(zip(x, mlr.coef_)))

x_pred = mlr.predict(x_train)
y_pred = mlr.predict(x_test)  

result = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
print(result)

print(mlr.score(x,y))