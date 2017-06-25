# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {0} data points with {1} variables each.".format(*data.shape)

from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt

plt.scatter(features.RM,prices, alpha = 0.5, c = prices)
plt.scatter([4,4,4],[8*10**5 + i for i in [10000,20000,30000]], s = [100]*3,c = 'black', marker = 'x')
plt.xlabel('X'); plt.ylabel('Y');
plt.show()

# =============== Regression Exercice ========================
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
X = np.random.randint(0,100,100).reshape(100,-1)
y = 2*X + 5 + np.random.normal(0,10,100).reshape(100,-1)
reg.fit(X,y)
print reg.coef_
print reg.intercept_

plt.scatter(X.reshape(-1),y)
plt.scatter(X.reshape(-1),reg.predict(X), c = 'r')
reg.score(X,y)


#=======================
#
#
# Regression and Classification programming exercises
#
#


#
#	In this exercise we will be taking a small data set and computing a linear function
#	that fits it, by hand.
#	

#	the data set

import numpy as np

sleep = [5,6,7,8,10]
scores = [65,51,75,75,86]


def compute_regression(sleep,scores):

    #	First, compute the average amount of each list

    avg_sleep = np.mean(sleep)
    avg_scores = np.mean(scores)

    #	Then normalize the lists by subtracting the mean value from each entry

    normalized_sleep = np.array(sleep) - avg_sleep
    normalized_scores = np.array(scores) - avg_scores

    #	Compute the slope of the line by taking the sum over each student
    #	of the product of their normalized sleep times their normalized test score.
    #	Then divide this by the sum of squares of the normalized sleep times.

    slope = (normalized_sleep*normalized_scores).sum()/float((normalized_sleep**2).sum())

    #	Finally, We have a linear function of the form
    #	y - avg_y = slope * ( x - avg_x )
    #	Rewrite this function in the form
    #	y = m * x + b
    #	Then return the values m, b
    m = slope
    b = avg_scores - avg_sleep*slope
    
    return m,b


if __name__=="__main__":
    m,b = compute_regression(sleep,scores)
    print "Your linear model is y={}*x+{}".format(m,b)
    
# ================== Poly Reg ===================
    
sleep = np.array([5,6,7,8,10,12,16])
scores = np.array([65,51,75,75,86,80,0])
n = len(sleep)
sleep = sleep.reshape(n,1)

from matplotlib import pyplot as plt
plt.scatter(sleep.reshape(-1),scores,c='b',marker ='x', linewidth = 10)

domain = np.linspace(3,20,30).reshape(30,-1)
X = np.ones((30,1))
plots = []
for k,c in zip([1,2,3],['r','g','y']):
    X = np.concatenate((domain*X[:,[0]],X),axis = 1)
    coef = np.polyfit(sleep.reshape(-1),scores,k)
    y_pred = X.dot(coef)
    plt_conf = plt.plot(domain,y_pred, c = c, label = str(k))
    plots.append(plt_conf)    

plt.legend(map(str,[1,2,3]))    
plt.show()    