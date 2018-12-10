# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:02:05 2018

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

n_person = 2000  #2000组参加 赌博
n_times =500    #硬币500次

t = np.arange(n_times)   #500次

#print(t)
steps = 2 * np.random.random_integers(0,1,(n_person,n_times)) - 1  #随机整数

#print(steps.shape)

amount = np.cumsum(steps,axis=1)    #计算每组的输赢总额
ad_amount = amount ** 2             #平方
mean_sd_amount = ad_amount.mean(axis=0) #所有组求平均

#print(amount.shape)
#print(ad_amount)
print(mean_sd_amount.shape)

plt.xlabel(r"$t$")
plt.ylabel(r"$\sqrt{(\delta x)^2 }$")
plt.plot(t, np.sqrt(mean_sd_amount), 'g.', t, np.sqrt(t), 'r-')