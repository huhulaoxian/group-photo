#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression


def classifier_data(y, lower_threshold = 6.0, upper_threshold = 6.0):
	for i, score in enumerate(y):
		if y[i] < lower_threshold:
			y[i] = 0
		elif y[i] >= upper_threshold:
			y[i] = 1

def load_data(classify_data):
	X = np.load('../../data/feature_vec/all_feature_new.npy')
	# X = X[:,7:]
	# X = preprocessing.scale(X)
	m = X.shape[0]
	n = X.shape[1]
	print(m,n)
	y = np.load('../../data/score_decimal_change.npy')
	if classify_data:
		classifier_data(y)

	trX = X
	trY = y
	
	return trX, trY 



if __name__ == '__main__':
    classify_data = False
    # trainX, trainY, testX, testY = load_data(classify_data)
    trainX, trainY = load_data(classify_data)
    if classify_data:
        clf = svm.SVC(kernel="linear",C = 1, gamma = 2,class_weight='balanced')
    else:
        clf = svm.SVR(kernel="linear",C = 1, gamma = 2)
        # clf = LinearRegression()
    selector = RFE(clf, 20, step=1)
    selector = selector.fit(trainX,trainY)
    print(selector.support_)
	
    print(selector.ranking_)
    np.save("svr_20.npy",selector.ranking_)
