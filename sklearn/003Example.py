# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:44:20 2018

@author: DELL
"""

from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()

image_and_lable = list(zip(digits.images,digits.target))

plt.figure(figsize=(8,6),dpi=100)

for index, (image,lable) in enumerate(image_and_lable[:8]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Digit: %i' % lable, fontsize=20)
    
print("shape of raw image data:{0}".format(digits.images.shape))
print("shape of data: {0}".format(digits.data.shape))

#模型训练
from sklearn.cross_validation import train_test_split
X_train,Xtest,Y_train,Ytest = train_test_split(digits.data, digits.target,
                                              test_size = 0.2,random_state = 2);                                         
from sklearn import svm
clf = svm.SVC(gamma=0.001,C=100.)
clf.fit(X_train, Y_train)

#模型测试
print(clf.score(Xtest,Ytest))

from sklearn.metrics import accuracy_score
Ypred = clf.predict(Xtest);
accuracy_score(Ytest, Ypred)

#查看
fig, axes = plt.subplots(4,4,figsize=(8,8))
fig.subplots_adjust(hspace=0.1,wspace=0.1)

for i,ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8,8),cmap=plt.cm.gray_r,
              interpolation='nearest')
    ax.text(0.05,0.05,str(Ypred[i]),fontsize=32,
            transform=ax.transAxes,
            color='green' if Ypred[i] == Ytest[i] else 'red')
    ax.text(0.8,0.05,str(Ypred[i]),fontsize=32,
            transform=ax.transAxes,
            color='black')
    ax.set_xticks([])
    ax.set_yticks([])

#保存模型参数
from sklearn.externals import joblib
joblib.dump(clf,'digits_svm.kpl')

clf = joblib.load('digits_svm.kpl')
Ypred = clf.predict(Xtest)
clf.score(Ytest,Ypred)