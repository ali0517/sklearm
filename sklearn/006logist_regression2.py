# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:19:24 2018

@author: aligo
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
import time
#数据
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print('data shape:{0}; no positive:{1}; no negative:{2}'.format(X.shape,
      y[y==1].shape, y[y==0].shape))

#数据分离
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#多项式拟合
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def polynomoal_model(degree=1,**kwarg):
    polynomial_feature = PolynomialFeatures(degree=degree, include_bias=False)
    logistic_reg = LogisticRegression(**kwarg)
    pipeline = Pipeline([("polynomial_feature",polynomial_feature),
                         ("logistic_reg",logistic_reg)])
    return pipeline

#模型训练 + 计算时间
start = time.clock()
model = polynomoal_model(degree=2,penalty='l1')  #默认是l2
model.fit(X_train,y_train)
print("time:{}".format(time.clock()-start))

#模型在训练集和测试集上的准确率
train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)
print('train score:{train_score};test score:{test_score}'.format(
        train_score=train_score,test_score=test_score))

#样本预测
y_pred = model.predict(X_test)
print('match: {0}/{1}'.format(np.equal(y_pred,y_test).shape[0], y_test.shape[0]))

#观察模型参数
logistic_reg = model.named_steps['logistic_reg']
print('model parameters shape:{0};non-zero element:{1}'.format(logistic_reg.coef_.shape,
      np.count_nonzero(logistic_reg.coef_)))
#由于二阶多项式的加入  30个特征变成了495个   95个是非0的