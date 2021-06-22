import numpy as np
from SaveNLoad import *
from sklearn import tree
import time
def DecisionTree(X_train, y_train, X_test, y_test):
    start = time.time()
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)
    end = time.time()
    Trainging_time = end - start
    save(clf, 'DecisionTree.sav')

    start = time.time()
    y_prediction = clf.predict(X_test)
    end = time.time()
    Test_Time = end - start
    accuracy = np.mean(y_prediction == y_test) * 100
    #print("The achieved accuracy using Decision Tree is " + str(accuracy))
    return accuracy, Trainging_time, Test_Time



