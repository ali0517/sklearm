# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 13:04:24 2018

@author: DELL
"""


# coding: utf-8

# # numpy基本操作
# 
# NumPy系统是Python的一种开源的数值计算扩展。这种工具可用来存储和处理大型矩阵。
# 更多操作参考http://www.runoob.com/numpy/numpy-tutorial.html
# 
# numpy reshape
#       
#       reshape()
#       revel()
#       
# numpy 加法 乘法 （数字）
# 
#       加法 乘法 （数组）      
#       点乘      （数组np.dot(a,b)）      
#       广播      （匹配维度，否则报错）
#       < = >       （bool判断）
#       
# numpy 内置函数
# 
#       cos()三角函数    
#       exp()指数
#       sqrt()平方根
#       
# numpy 统计功能
# 
#       sum()
#       mean()
#       min() max()
#       argmin()
#       argmax()  返回索引
#       std()    标准差
#       
# numpy 多维度 行列
# 
#       axis = 0
#       axis = 1
#       
# numpy 数组排序功能
# 
#       np.sort()
#       np.argsout()
#       
# numpy 求解多项式   多项式拟合
#     
# numpy 文件操作

# ### numpy reshape

import numpy as np

a = np.arange(12)
print(a)

b = a.reshape(3,4)
print(b)

c = b.ravel()
print(c)


# ### numpy 加法 乘法 （数字）

# In[ ]:
a = np.arange(6)
print(a+5)

# In[ ]:
b = a.reshape(2,3)
print(b)
c = np.ones((3,2))

d = np.dot(b,c)
print(d)

# In[ ]:
e = np.arange(3)
f = np.arange(2)

print(b + e)
print(b + f.reshape(2,1))

# In[ ]:
g = [[2,3]]
print(f > g)

# ### numpy 内置函数
# In[ ]:
a = np.arange(6)

print(np.cos(a))
print(np.exp(a))
print(np.sqrt(a))

# ### numpy 统计功能
# In[ ]:
a = np.random.randint(1,8,20)
print(a)

print(a.sum())
print(a.mean())
print(a.min())
print(a.argmax())

# ### numpy 多维度 行列
# In[ ]:
b = np.random.randint(1,8,(5,6))
print(b)

print(b.sum(axis=1))
print(b.std(axis=1))

# ### numpy 数组排序功能
# In[ ]:
c = np.sort(b,axis=1)
d = np.argsort(b,axis=0)

print(c)
print(d)

# ### numpy 求解多项式   多项式拟合
# In[ ]:
p = np.poly1d([1,-4,3])   #二项式的系数

print(p(1))

print(p.roots)      #多项式的根
print(p.order)      #多项式阶数
print(p.coeffs)     #多项式系数


# %matplotlib inline
# import matplotlib.pyplot as plt
# 
# n_dots = 20
# n_order =3
# 
# x = np.linspace(0,1,20)  #0,1之间的20个点
# y = np.sqrt(x) + 0.2*np.random.rand(n_dots)
# 
# p = np.poly1d(np.polyfit(x,y,n_order))
# print(p.coeffs)
# 
# t = np.linspace(0,1,200)
# plt.plot(x,y,'ro',t,p(t),'-')

# ### 计算pi

# In[ ]:
n_dots = 1000000
x = np.random.random(n_dots)
y = np.random.random(n_dots)

distance = np.sqrt(x**2 + y**2)

in_circle = distance[distance<1]
print(in_circle.shape)

pi = 4*float(len(in_circle)) / n_dots
print(pi)

# ### numpy 文件操作
