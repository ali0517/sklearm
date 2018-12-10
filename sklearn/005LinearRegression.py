# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:53:28 2018

@author: DELL
"""

from sklearn.linear_model import LinearRegression      #线性回归
from sklearn.preprocessing import PolynomialFeatures   #多项式类
from sklearn.pipeline import Pipeline                  #管道 串联两个类

import numpy as np
import matplotlib.pyplot as plt
#生成200个在 点 【-2pi，2pi】
n_dot = 200

X = np.linspace(-2*np.pi, 2*np.pi, n_dot)
Y = np.sin(X) + 0.5*np.random.rand(n_dot) - 0.1  #加一些随机噪声
print(X.shape,Y.shape)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(X,Y)

X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
print(X.shape,Y.shape)

#创建一个多项式拟合
def polynomial_model(degree = 1):
    polynomial_features = PolynomialFeatures(degree = degree,
                                            include_bias=False)
    linear_regression = LinearRegression()
    
    #pipeline是一个流水线  先增加多项式阶数， 然后再用线性回归拟合
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline

from sklearn.metrics import mean_squared_error

degress = [2,3,5,10]
results = []

for d in degress:
    model = polynomial_model(degree=d)
    model.fit(X,Y)
    train_score = model.score(X,Y)
    mse = mean_squared_error(Y,model.predict(X))
    results.append({"model":model, "degree":d, "score":train_score, "mse":mse})
for r in results:
    print("degree:{}; train_score:{}; mean squared error:{}".format(
            r["degree"], r["score"], r["mse"]))
    
from matplotlib.figure import SubplotParams

plt.figure(figsize=(12,6), dpi=200, subplotpars = SubplotParams(hspace=0.3))

for i, r in enumerate(results):
    fig = plt.subplot(2, 2, i+1)
    plt.xlim(-8, 8)
    plt.title("LinearRegression degree={}".format(r["degree"]))
    plt.scatter(X, Y, s=5, c='b', alpha=0.5)
    plt.plot(X, r["model"].predict(X), 'r-')   
