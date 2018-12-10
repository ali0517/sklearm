# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:25:18 2018

@author: aligo
"""

import matplotlib.pyplot as plt
import numpy as np
from aligo_alg import Gini_impurity


X = np.linspace(0,1,500)
Y = Gini_impurity.gini_impurity(X)
plt.xlabel("P(x)")

plt.ylabel("P(x)(1-P(x))")
plt.plot(X,Y)