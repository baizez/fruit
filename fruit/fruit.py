from sklearn import svm
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np

#np.set_printoptions(threshold=np.NaN) 
data = sio.loadmat('data/fruit0326.mat')
Xtrain=data['CSH_train_baseline']
Xtest=data['CSH_test_baseline']
YtrainLabel=data['TH_train']
Ytrain=list()
for row in YtrainLabel:
    Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

clf=svm.SVC(kernel='linear',decision_function_shape='ovo')
res=clf.fit(Xtrain,Ytrain)
print(res)
print(clf.predict(Xtest))