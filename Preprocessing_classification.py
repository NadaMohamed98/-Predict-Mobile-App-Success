from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from Preprocessing_classification import *
from sklearn import linear_model
from Logistic_Regression import *
from KNN import *
from DecisionTree import *
from SVM import *
from matplotlib import pyplot
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd


def select_features(X_train, y_train, X_test):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


def pre_process2(file_name):
    # Loading data
    # data = pd.read_csv('AppleStore_training_classification.csv')
    data = pd.read_csv(file_name)

    # Handling missing values(numbers)
    missing_cols = {'size_bytes', 'price', 'rating_count_tot', 'rating_count_ver', 'vpp_lic',
                    'sup_devices.num', 'ipadSc_urls.num', 'lang.num'}
    for f in missing_cols:
        med = data[f].median()
        data[f] = data[f].fillna(med)

    # Handling missing values(categories)
    missing_cols_cat = {'currency', 'ver', 'cont_rating', 'prime_genre'}
    for f in missing_cols_cat:
        data[f] = data[f].fillna('UnKnown')

    # Label_Encoding
    string_cols = {'currency', 'ver', 'cont_rating', 'prime_genre', 'rate'}
    for c in string_cols:
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))

    # Feature Selection
    """x = data.iloc[:, 2:14]
    y = data['rate']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True)

    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()"""

    x = data[['rating_count_ver', 'ipadSc_urls.num', 'lang.num']]
    y = data['rate']

    # scale = StandardScaler()
    # x = scale.fit_transform(x)
    x = preprocessing.normalize(x, axis=0)

    # Store Features and Label
    # x = data[['ipadSc_urls.num', 'lang.num']]

    # Feature_Scaling(Normalization)
    # x = preprocessing.normalize(x, axis=0)
    # y = preprocessing.normalize(y, axis=0)

    # x = x.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x = min_max_scaler.fit_transform(x)

    return x, y


"""#Handeling missing values(numbers)
    missing_cols = {'size_bytes', 'price', 'rating_count_tot', 'rating_count_ver', 'vpp_lic',
                   'sup_devices.num', 'ipadSc_urls.num', 'lang.num'}
    for f in missing_cols:
        med = data[f].median()
        data[f] = data[f].fillna(med)


    #Handeling missing values(categories)
    missing_cols_cat = {'currency', 'ver', 'cont_rating', 'prime_genre'}
    for f in missing_cols_cat:
        data[f] = data[f].fillna('UnKnown')


    #Store Features and Label
    x = data.iloc[:, 2:14]
    y = data['rate']

    #Label_Encoding
    string_cols = {'currency', 'ver', 'cont_rating', 'prime_genre'}
    for c in string_cols:
        lbl = LabelEncoder()
        lbl.fit(list(x[c].values))
        x[c] = lbl.transform(list(x[c].values))

    #One_Hot_Encoding_
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [8])], remainder='passthrough')
    x = np.array(columnTransformer.fit_transform(x), dtype=np.str)

    #Feature_Scaling(Standardization)
    # x = preprocessing.scale(x)

    #Feature_Scaling(Normalization)
    x = preprocessing.normalize(x, axis=0)"""
