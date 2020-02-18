#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold


def classifier_data(y, lower_threshold = 6.0, upper_threshold = 6.0):
	for i, score in enumerate(y):
		if y[i] < lower_threshold:
			y[i] = 0
		elif y[i] >= upper_threshold:
			y[i] = 1


def load_data(classify_data,feature):
    X = np.load(feature)	

	# X = preprocessing.scale(X)
    print(X)
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
    classify_data = True
    # trainX, trainY, testX, testY = load_data(classify_data)
    X1, y1 = load_data(classify_data,"../../data/feature_vec/filter_svc_20.npy")
    if classify_data:
        clf = svm.SVC(C = 1, gamma = 2,class_weight='balanced',probability=True)
    else:
        clf = svm.SVR(C = 1, gamma = 3.7)

    fig = plt.figure() 
    axes = fig.add_subplot(111)


    k_fold = KFold(n_splits=5, shuffle=True, random_state=12345)
    predictor = clf
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        # axes.plot(recall, precision, label=lab)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'with GF model AUC=%.4f' % (auc(recall, precision))
    axes.plot(recall, precision, label=lab, lw=2)

    y_real1 = []
    y_proba1 = []
    X2, y2 = load_data(classify_data,"../../data/feature_vec/filter_svc_without_20.npy")
    for i, (train_index, test_index) in enumerate(k_fold.split(X2)):
        Xtrain, Xtest = X2[train_index], X2[test_index]
        ytrain, ytest = y2[train_index], y2[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        # axes.plot(recall, precision, label=lab)
        y_real1.append(ytest)
        y_proba1.append(pred_proba[:,1])

    y_real1 = np.concatenate(y_real1)
    y_proba1 = np.concatenate(y_proba1)
    precision, recall, _ = precision_recall_curve(y_real1, y_proba1)
    lab = 'without GF model AUC=%.4f' % (auc(recall, precision))
    axes.plot(recall, precision, label=lab, lw=2)

    X3, y3 = load_data(classify_data,"../../data/feature_vec/onlyface_warpper5.npy")
    y_real2 = []
    y_proba2 = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X3)):
        Xtrain, Xtest = X3[train_index], X3[test_index]
        ytrain, ytest = y3[train_index], y3[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
        # axes.plot(recall, precision, label=lab)
        y_real2.append(ytest)
        y_proba2.append(pred_proba[:,1])

    y_real2 = np.concatenate(y_real2)
    y_proba2 = np.concatenate(y_proba2)
    precision, recall, _ = precision_recall_curve(y_real, y_proba2)
    lab = 'only GF model AUC=%.4f' % (auc(recall, precision))
    axes.plot(recall, precision, label=lab, lw=2)
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    plt.title("Precision-Recall")
    axes.legend(loc='lower left', fontsize='small')
    plt.show()


