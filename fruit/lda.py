import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io as sio  

data = sio.loadmat('data/fruit0329.mat')
Xtrain=data['CSH_train_baseline']
Xtest=data['CSH_test_baseline']
YtrainLabel=data['TH_train2']
Ytrain=list()
for row in YtrainLabel:
    Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

lda = LinearDiscriminantAnalysis(n_components=2)

lda.fit(Xtrain,Ytrain)
X_train=lda.transform(Xtrain)
Ytest=lda.predict(Xtest)
X_test=lda.transform(Xtest)
plt.scatter(X_train[:, 0], X_train[:, 1],marker='o',c=Ytrain)
#plt.scatter(X_test[:, 0], X_test[:, 1],marker='x',c=Ytest)
plt.show()


XtrainAll=data['CSH_all_baseline']
YtrainLabelAll=data['TH_all2']
YtrainAll=list()
for row in YtrainLabelAll:
    YtrainAll.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

lda = LinearDiscriminantAnalysis(n_components=2)

lda.fit(XtrainAll,YtrainAll)
X_trainAll=lda.transform(XtrainAll)
dataNew='data/fruit0329_Lda'
sio.savemat(dataNew,{'CSH_tran_baseline_lda':X_train,'CSH_test_baseline_lda':X_test,'CSH_all_baseline_lda':X_trainAll})
plt.scatter(X_trainAll[:, 0], X_trainAll[:, 1],marker='o',c=YtrainAll)
plt.show()