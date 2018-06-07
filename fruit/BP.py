from sklearn.neural_network import MLPClassifier
from fruit import Fruit
import numpy as np
from sklearn.metrics import accuracy_score


fruit_train=Fruit(task="train")
fruit_test=Fruit(task="test")
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(14,14,30), random_state=1)

clf.fit(fruit_train.images,fruit_train.labels)

ypred=clf.predict(fruit_test.images)
print(accuracy_score(ypred,fruit_test.labels))

print(ypred)
