from sklearn import svm
import csv


with open('csv/fruit0326_FSH_train_1_0_.csv') as traindata:
    train = csv.reader(traindata)
    for row in train:
        print(row)