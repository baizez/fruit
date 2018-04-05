from sklearn import svm
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np
import time

data = sio.loadmat('data/fruit0329.mat')
Xtrain=data['CSH_train_baseline']
Xtest=data['CSH_test_baseline']
YtrainLabel=data['TH_train2']
Ytrain=list()
for row in YtrainLabel:
    Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

clf=svm.SVC(kernel='linear',decision_function_shape='ovo')

start = time.clock()
res=clf.fit(Xtrain,Ytrain)
print(time.clock() - start)

print(res)
print(clf.predict(Xtest))