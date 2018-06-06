import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.io as sio  

def lda(dataName,label):
    data = sio.loadmat('data/fruit0329.mat')
    Xtrain=data[dataName]
    YtrainLabel=data[label]
    Ytrain=list()
    for row in YtrainLabel:
        Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(Xtrain,Ytrain)
    X_train=lda.transform(Xtrain)
    plt.figure(dataName)
    plt.scatter(X_train[:, 0], X_train[:, 1],marker='o',c=Ytrain)
    plt.title(dataName)
    plt.savefig('result/'+dataName+'.png')
    plt.show()
    return X_train
    
if __name__ == '__main__':
    datalist=['CSH_train_baseline','CSH_all_baseline','CSH_all_RR']
    datalabel={'CSH_train_baseline':'TH_train2','CSH_all_baseline':'TH_all','CSH_all_RR':'TH_all'}
    ldadict=dict()
    ldaMat='data/fruit0329_Lda'
    for dataname in datalist:
        trained=lda(dataname,datalabel[dataname])
        ldadict[dataname+'_lda']=trained
    sio.savemat(ldaMat,ldadict)