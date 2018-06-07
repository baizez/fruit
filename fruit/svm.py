from sklearn.svm import SVC
from fruit import Fruit
import numpy as np
from sklearn.metrics import accuracy_score

def svmFit(data,label):
    Ytrain=list()
    for row in label:
        Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

    clf=SVC(kernel='linear',decision_function_shape='ovo')
    clf=clf.fit(data,Ytrain)
    return clf

def svmPredict(clf,data,label):
    Ytrain=list()
    for row in label:
        Ytrain.append(row[0]*1+row[1]*2+row[2]*3+row[3]*4)

    Ypre = clf.predict(data)
    print(Ypre)
    return accuracy_score(Ypre,Ytrain)
    
if __name__ == '__main__':
    fruit_train=Fruit(task="train")
    fruit_test=Fruit(task="test")
    model=svmFit(fruit_train.images,fruit_train.labels)
    print(svmPredict(model,fruit_test.images,fruit_test.labels))
