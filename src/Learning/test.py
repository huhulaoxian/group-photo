#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def load_data(classify_data):
    X = np.load('../../data/feature_vec/compare1_rf_all.npy')	

	# X = preprocessing.scale(X)
    m = X.shape[0]
    n = X.shape[1]
    print(m,n)
    trX = X
	
    return trX

if __name__ == '__main__':
    classify_data = False
    testX= load_data(classify_data)
    clf = pickle.load(open( "rf_all_20.p", "rb" ))
    prediction = clf.predict(testX)
    print(prediction)