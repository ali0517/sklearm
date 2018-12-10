# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:52:48 2018

@author: DELL
"""
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
Y = boston.target
print(boston.data.shape)
print(boston.feature_names)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state = 3)

import time 
from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=True)

start = time.clock()
model.fit(X_train,Y_train)
train_score = model.score(X_train, Y_train)
cv_score = model.score(X_test,Y_test)

print('elaspe:{}; train_sorce:{}; cv_score:{}'.format(time.clock()-start,
      train_score,cv_score))

#使用二阶多项式拟合     三阶 四阶会有过拟合现象
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline

model = polynomial_model(degree=2)

start = time.clock()
model.fit(X_train, Y_train)

train_score = model.score(X_train, Y_train)
cv_score = model.score(X_test, Y_test)
print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.clock()-start, train_score, cv_score))

#学习曲线
#from common.utils import plot_learning_curve
#from sklearn.model_selection import ShuffleSplit
#
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#plt.figure(figsize=(18, 4))
#title = 'Learning Curves (degree={0})'
#degrees = [1, 2, 3]
#
#start = time.clock()
#plt.figure(figsize=(18, 4), dpi=200)
#for i in range(len(degrees)):
#    plt.subplot(1, 3, i + 1)
#    plot_learning_curve(plt, polynomial_model(degrees[i]), title.format(degrees[i]), X, Y, ylim=(0.01, 1.01), cv=cv)
#
#print('elaspe: {0:.6f}'.format(time.clock()-start))