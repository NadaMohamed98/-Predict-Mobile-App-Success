from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def pre_process(filename):
    #data = pd.read_csv('AppleStore_training.csv')
    # drop rows with missing values
    data = pd.read_csv(filename)
    data.dropna(how='any', inplace=True)

    # Store Features and Label
    #App_data = data.iloc[:, :]

    # Label_Encoding
    string_cols = {'currency', 'ver', 'cont_rating', 'prime_genre'}
    for c in string_cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))

    #Correlation
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    sns.heatmap(data[top_corr_features].corr(), annot=True)
    plt.show()

    x = data[['user_rating_ver', 'ipadSc_urls.num', 'lang.num']]
    y = data['user_rating']
    # One_Hot_Encoding_
    columnTransformer = ColumnTransformer(
        [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    x = np.array(columnTransformer.fit_transform(x), dtype=np.str)

    # Feature_Scaling(Standardization)
    # x = preprocessing.scale(x)

    # Feature_Scaling(Normalization)
    x = preprocessing.normalize(x, axis=0)
    return x, y


def Plot(x_train, x_test, y_train, prediction):
    # Plotting
    for i in range(len(x_train[0]) - 1):
        plt.scatter(np.array(x_train)[:, i], y_train)
    plt.xlabel('f', fontsize=20)
    plt.ylabel('user rating', fontsize=20)
    plt.plot(x_test, prediction, color='red', linewidth=3)
    plt.show()
