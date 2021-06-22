from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import time
from SaveNLoad import *
from sklearn import metrics
import numpy as np


def PolyReg(X_train, y_train, x_test, y_test):
    start = time.time()
    poly_features = PolynomialFeatures(degree=3)

    XTr = poly_features.fit_transform(X_train)

    poly_model = linear_model.Ridge(alpha=1)
    poly_model.fit(XTr, y_train)
    save(poly_model, 'PolyRegression.sav')

    end = time.time()
    Training_time = end - start

    start = time.time()
    y_prediction = poly_model.predict(poly_features.fit_transform(x_test))
    MSE = metrics.mean_squared_error(y_test, y_prediction)
    r2_score = metrics.r2_score(y_test, y_prediction)
    #accuracy = np.mean(y_prediction == y_test) * 100
    end = time.time()
    Testing_time = end - start
    # print('Co-efficient of linear regression',cls.coef_)
    # print('Intercept of linear regression model',cls.intercept_)
    return MSE, r2_score, Training_time, Testing_time

