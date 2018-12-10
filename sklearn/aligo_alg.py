# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:26:34 2018

@author: aligo
"""
import numpy as np
import matplotlib.pyplot as plt

#计算熵
class Entropy(object):
    def __init__(self,name):
        self.name = name
    
    def entropy(X):
        Y = -X * np.log2(X)
        return Y

class Gini_impurity(object):
    def __init__(self,name):
        self.name = name
        
    def gini_impurity(X):
        Y = X * (1-X)
        return Y
    
if __name__ == "main":
    
    X = np.linspace(0,1,500)
    Y = Entropy.entropy(X)    
    plt.plot(X,Y)