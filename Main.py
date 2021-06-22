from sklearn.model_selection import train_test_split
from Preprocessing_classification import *
from Logistic_Regression import *
from KNN import *
from DecisionTree import *
from SVM import *
import matplotlib.pyplot as plt
from LinearReg import *
from PolynomialReg import *
from TestScript import *

# M1
# x, y = pre_process('AppleStore_training.csv')
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

# Linear reg
# linear_reg_mse, linear_reg_r2_score, Training_time, Test_time = DoLinearReg(x_train, y_train, x_test, y_test)
# print(" Linear reg: MSE: ", linear_reg_mse)
# print(" Linear reg: r2 score: ", linear_reg_r2_score)
#
# # Polynomial reg
# poly_reg_mse, poly_reg_r2_score, Training_time, Test_time = PolyReg(x_train, y_train, x_test, y_test)
# print(" poly reg: MSE: ", poly_reg_mse)
# print(" poly reg: r2 score: ", poly_reg_r2_score)
'''
# M2
x, y = pre_process2("AppleStore_training_classification.csv")
Accuracies = []
Training_Times = []
Testing_Times = []

# splitting data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

# Decision Tree
Accuracy, train_time, test_time = DecisionTree(X_train, y_train, X_test, y_test)
print("Decision Tree : Accuracy ", Accuracy)
print("Decision Tree : Training_Time ", train_time)
print("Decision Tree : Test_Time ", test_time)

Accuracies.append(Accuracy)
Training_Times.append(train_time)
Testing_Times.append(test_time)

# KNN
Accuracy, train_time, test_time = Knn(X_train, y_train, X_test, y_test)
print("KNN : Accuracy ", Accuracy)
print("KNN : Training_Time ", train_time)
print("KNN : Test_Time ", test_time)

Accuracies.append(Accuracy)
Training_Times.append(train_time)
Testing_Times.append(test_time)

# Logistic Regression
Accuracy1, Accuracy2, Training_Time1, Training_Time2, Test_time1, Test_time2 = Logistic_reg(X_train, y_train, X_test,
                                                                                            y_test)
print("Logistic Regression Ordinary : Accuracy ", Accuracy1)
print("Logistic Regression Ordinary : Training_Time ", Training_Time1)
print("Logistic Regression Ordinary : Test_Time ", Test_time1)
print("Logistic Regression OnevsAll : Accuracy ", Accuracy2)
print("Logistic Regression OnevsAll : Training_Time ", Training_Time2)
print("Logistic Regression OnevsAll : Test_Time ", Test_time2)

Accuracies.append(Accuracy1)
Training_Times.append(Training_Time1)
Testing_Times.append(Test_time1)
Accuracies.append(Accuracy2)
Training_Times.append(Training_Time2)
Testing_Times.append(Test_time2)

# SVM
Accuracy1, Accuracy2, Training_Time1, Training_Time2, Test_time1, Test_time2 = SVM(X_train, y_train, X_test, y_test)
print("SVM ovr : Accuracy ", Accuracy1)
print("SVM ovr : Training_Time ", Training_Time1)
print("SVM ovr : Test_Time ", Test_time1)
print("SVM ovo : Accuracy ", Accuracy2)
print("SVM ovo : Training_Time ", Training_Time2)
print("SVM ovo : Test_Time ", Test_time2)

Accuracies.append(Accuracy1)
Training_Times.append(Training_Time1)
Testing_Times.append(Test_time1)
Accuracies.append(Accuracy2)
Training_Times.append(Training_Time2)
Testing_Times.append(Test_time2)

# accuracy fig
Models = ["Decision Tree", "KNN", "Logistic Regression Ordinary", "Logistic Regression OnevsAll", "SVM ovr", "SVM ovo"]
fig = plt.figure(figsize=(10, 5))
plt.bar(Models, Accuracies, color='purple', width=0.4)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy Chart")
plt.show()

# training time fig
fig = plt.figure(figsize=(10, 5))
plt.bar(Models, Training_Times, color='pink', width=0.4)
plt.xlabel("Model")
plt.ylabel("Training Time")
plt.title("Training Chart")
plt.show()

# test time fig
fig = plt.figure(figsize=(10, 5))
plt.bar(Models, Testing_Times, color='Gray', width=0.4)
plt.xlabel("Model")
plt.ylabel("Test Time")
plt.title("Test Chart")
plt.show()
'''
# Load saved models
saved_models()

