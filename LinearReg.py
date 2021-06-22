from sklearn import linear_model
from sklearn import metrics
import time
from SaveNLoad import *
import numpy as np


def DoLinearReg(x_train, y_train, x_test, y_test):
    start = time.time()
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    save(model, 'LinearRegression.sav')

    end = time.time()
    Training_time = end-start

    start = time.time()
    y_prediction = model.predict(x_test)
    MSE = metrics.mean_squared_error(y_test, y_prediction)
    r2_score = metrics.r2_score(y_test, y_prediction)
    #accuracy = np.mean(y_prediction == y_test) * 100
    end = time.time()
    Testing_time = end - start

    # print('Co-efficient of linear regression',cls.coef_)
    # print('Intercept of linear regression model',cls.intercept_)
    return MSE, r2_score, Training_time, Testing_time




