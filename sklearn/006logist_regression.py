# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:19:24 2018

@author: aligo
"""
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print('data shape:{0}; no positive:{1}; no negative:{2}'.format(X.shape,
      y[y==1].shape, y[y==0].shape))

print(cancer.data[33])
#一共569条数据   每条又30个特征    正特征y=1有357个  负特征y=0有212个
print(cancer.feature_names)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2')
model.fit(X_train,y_train)

train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)
print('train score:{train_score};test score:{test_score}'.format(
        train_score=train_score,test_score=test_score))

#样本预测
y_pred = model.predict(X_test)


print('match: {0}/{1}'.format(np.equal(y_pred,y_test).shape[0], y_test.shape[0]))


##瞎写的
#aa = model.predict(X_test)
#aa.reshape(1,-1)
#i = 55
#print('第{2}个样本属于：{0}; 标记的类别：{1}'.format(aa[i], y_test[i], i))
#
#print(X_test.shape)
#print(aa.shape)
#
#b = X_test[:33,:]
#b.reshape(1,-1)
##print(b)
#
#aaa = model.predict(b)
#print(aaa)


#概率数据 predict_proba    【0的概率，1的概率】
y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba[0])

