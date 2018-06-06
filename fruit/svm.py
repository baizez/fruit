from sklearn.svm import SVC
import scipy.io as sio  
import numpy as np
from sklearn.metrics import accuracy_score

def svmFit(dataName,label):
    data = sio.loadmat('data/fruit0329.mat')
    Xtrain=data[dataName]
    YtrainLabel=data[label]
    Ytrain=list()
    for row in YtrainLabel:
        Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

    clf=SVC(kernel='linear',decision_function_shape='ovo')
    clf=clf.fit(Xtrain,Ytrain)
    return clf

def svmPredict(clf,dataName,label):
    data = sio.loadmat('data/fruit0329.mat')
    Xtrain=data[dataName]
    YtrainLabel=data[label]
    Ytrain=list()
    for row in YtrainLabel:
        Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

    Ypre = clf.predict(Xtrain)
    return accuracy_score(Ytrain,Ypre)
    
if __name__ == '__main__':
    trainlist=['CSH_train_baseline','CSH_all_baseline','CSH_train_RR',]
    testlist=['CSH_test_baseline','CSH_test_RR'] 
    datalabel={'CSH_train_baseline':'TH_train2','CSH_test_baseline':'TH_test2',
               'CSH_all_baseline':'TH_all2','CSH_train_RR':'TH_train2','CSH_test_RR':'TH_test2'}
    model=svmFit(trainlist[0],datalabel[trainlist[0]])
    print(svmPredict(model,testlist[0],datalabel[testlist[0]]))
    model=svmFit(trainlist[2],datalabel[trainlist[2]])
    print(svmPredict(model,testlist[1],datalabel[testlist[1]]))