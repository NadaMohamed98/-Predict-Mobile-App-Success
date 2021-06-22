from sklearn.svm import SVC
from SaveNLoad import *
import time


def SVM(X_train, y_train, X_test, y_test):
    # training a linear SVM classifier
    start = time.time()
    svm_model_linear_ovr = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovr').fit(X_train, y_train)
    # svm_predictions = svm_model_linear_ovr.predict(X_test)
    # model accuracy for X_test
    end = time.time()
    Training_Time1 = end - start
    save(svm_model_linear_ovr, 'svm_model_linear_ovr.sav')
    start = time.time()
    accuracy1 = svm_model_linear_ovr.score(X_test, y_test) * 100
    end = time.time()
    Test_Time1 = end - start
    # print('One VS Rest SVM accuracy: ' + str(accuracy1 * 100))

    start = time.time()
    svm_model_linear_ovo = SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    end = time.time()
    Training_Time2 = end - start
    start = time.time()
    # svm_predictions = svm_model_linear_ovo.predict(X_test)
    accuracy2 = svm_model_linear_ovo.score(X_test, y_test) * 100
    end = time.time()
    Test_Time2 = end - start
    save(svm_model_linear_ovo, 'svm_model_linear_ovo.sav')
    # print(confusion_matrix(y_test, svm_predictions))
    # print(classification_report(y_test, svm_predictions, zero_division=0))
    # model accuracy for X_test
    # accuracy = svm_model_linear_ovo.score(X_test, y_test)
    # print('One VS One SVM accuracy: ' + str(accuracy * 100))
    return accuracy1, accuracy2, Training_Time1, Training_Time2, Test_Time1, Test_Time2
