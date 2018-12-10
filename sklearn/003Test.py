# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:31:19 2018

@author: DELL
"""
from sklearn.externals import joblib
from sklearn import datasets

digits = datasets.load_digits()

from sklearn.cross_validation import train_test_split
X_train,Xtest,Y_train,Ytest = train_test_split(digits.data, digits.target,
                                              test_size = 0.2,random_state = 2); 
                                               
clf = joblib.load('digits_svm.kpl')
Ypred = clf.predict(Xtest)
clf.score(Ytest,Ypred)