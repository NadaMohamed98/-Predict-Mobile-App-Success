from sklearn.model_selection import train_test_split
from sklearn import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn import linear_model
from Preprocessing_classification import *
from sklearn.preprocessing import PolynomialFeatures
from DecisionTree import *
from SVM import *
import matplotlib.pyplot as plt
from PreProcessing import *


def saved_models():
    # M1
    # Loading Data
    mse_list = []
    r2_score_list = []
    x, y = pre_process('AppleStore_testing.csv')
   # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

    # Linear Reg
    LinearReg = load('LinearRegression.sav')
    y_prediction = LinearReg.predict(x)
    MSE = metrics.mean_squared_error(y, y_prediction)
    print('Linear reg mse', MSE)
    r2_score = metrics.r2_score(y, y_prediction)
    print('Linear reg r2score', r2_score)
    mse_list.append(MSE)
    r2_score_list.append(r2_score)

    # accuracy = np.mean(y_prediction == y_test) * 100
    # accuracies.append(accuracy)

    # Polynomaial Reg
    poly_features = PolynomialFeatures(degree=3)
    PolynomaialReg = load('PolyRegression.sav')
    y_prediction = PolynomaialReg.predict(poly_features.fit_transform(x))
    MSE = metrics.mean_squared_error(y, y_prediction)
    print('poly reg mse', MSE)
    r2_score = metrics.r2_score(y, y_prediction)
    print('poly reg r2score', r2_score)
    mse_list.append(MSE)
    r2_score_list.append(r2_score)
    # accuracy = np.mean(y_prediction == y_test) * 100
    # accuracies.append(accuracy)

    # plot
    models = ["Linear Regression", "Polynomial Regression"]
    # MSE
    fig = plt.figure(figsize=(10, 5))
    plt.bar(models, mse_list, color='purple', width=0.4)
    plt.xlabel("Model")
    plt.ylabel("MSE")
    plt.title("MSE Chart")
    plt.show()

    # r2_score
    fig = plt.figure(figsize=(10, 5))
    plt.bar(models, r2_score_list, color='purple', width=0.4)
    plt.xlabel("Model")
    plt.ylabel("R2_Score")
    plt.title("R2_Score Chart")
    plt.show()

    # -------------------------------------------------------------------------------------

    # M2
    # Loading data
    x, y = pre_process2('AppleStore_testing_classification.csv')
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)
    accuracies = []

    # DecisionTree
    decision_tree_model = load('DecisionTree.sav')
    y_prediction = decision_tree_model.predict(x)
    accuracy = np.mean(y_prediction == y) * 100
    print('d t accuracy: ', accuracy)
    accuracies.append(accuracy)


    # Logistic Regression with multinomial
    logistic_reg_model = load('Logistic_reg_ordinary.sav')
    y_prediction = logistic_reg_model.predict(x)
    accuracy = np.mean(y_prediction == y) * 100
    print('log reg1 accuracy: ', accuracy)
    accuracies.append(accuracy)

    # Logistic Regression with OneVersusAll
    logistic_reg_ovo_model = load('Logistic_reg_ordinary.sav')
    y_prediction = logistic_reg_ovo_model.predict(x)
    accuracy = np.mean(y_prediction == y) * 100
    print('log reg2 accuracy: ', accuracy)
    accuracies.append(accuracy)

    # SVM_1
    svm_linear_model = load('svm_model_linear_ovr.sav')
    y_prediction = svm_linear_model.predict(x)
    accuracy = np.mean(y_prediction == y) * 100
    print('svm 1 accuracy: ', accuracy)
    accuracies.append(accuracy)

    # SVM_2
    svm_linear_ovo_model = load('svm_model_linear_ovo.sav')
    y_prediction = svm_linear_ovo_model.predict(x)
    accuracy = np.mean(y_prediction == y) * 100
    print('svm 2 accuracy: ', accuracy)
    accuracies.append(accuracy)



    # plot
    models = ["Decision Tree", "Logistic Regression Ordinary", "Logistic Regression OnevsAll", "SVM ovr",
              "SVM ovo"]
    fig = plt.figure(figsize=(10, 5))
    plt.bar(models, accuracies, color='purple', width=0.4)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Chart")
    plt.show()
