from sklearn import linear_model
import numpy as np
from SaveNLoad import *
import time

def Logistic_reg(X_train, y_train , x_test , y_test):
    # Logistic Regression with multinomial
    start = time.time()
    lm1 = linear_model.LogisticRegression(penalty='l2',C=100, solver='lbfgs',max_iter = 1000, multi_class='multinomial')
    lm1.fit(X_train, y_train)
    end = time.time()
    Training_Time1 = end - start
    save(lm1, 'Logistic_reg_ordinary.sav')
    start = time.time()
    y_prediction = lm1.predict(x_test)
    end = time.time()
    Test_Time1 = end - start
    accuracy1 = np.mean(y_prediction == y_test) * 100

    # Logistic Regression with OneVersusAll
    start = time.time()
    lm2 = linear_model.LogisticRegression(penalty='l1', multi_class='ovr', solver='liblinear', C=50, max_iter=1000)
    lm2.fit(X_train, y_train)
    end = time.time()
    Training_Time2 = end - start
    save(lm2, 'Logistic_reg_OneVersusAll.sav')
    start = time.time()
    y_prediction = lm2.predict(x_test)
    end = time.time()
    Test_Time2 = end - start
    accuracy2 = np.mean(y_prediction == y_test) * 100
    return accuracy1, accuracy2, Training_Time1, Training_Time2, Test_Time1, Test_Time2
