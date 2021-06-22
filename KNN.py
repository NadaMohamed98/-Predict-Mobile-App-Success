import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import math
from SaveNLoad import *
import time


def Knn(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    error = []
    bestModel = 0
    TrainingTime = 0
    TestTime = 0
    Accuracy = 0
    MinError = math.inf
    # Calculating error for K values between 1 and 100
    for i in range(1, 100):
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        end = time.time()
        modelTrainingTime = end - start
        start = time.time()
        y_pred = knn.predict(X_test)
        end = time.time()
        modelTestTime = end - start
        Error = np.mean(y_pred != y_test)
        error.append(Error)
        if Error < MinError:
            MinError = Error
            bestModel = knn
            TrainingTime = modelTrainingTime
            TestTime = modelTestTime
            Accuracy = metrics.accuracy_score(y_test, y_pred) * 100
            #print("Accuracy: ", Accuracy)
            #print("k: ", i)


    #    print("Error:", Error)
    #    print("Accuracy:", metrics.accuracy_score(y_test, y_pred) *100)
    # print("Min Error:", MinError)
    #save(bestModel, 'Knn.sav')
    return Accuracy, TrainingTime, TestTime
