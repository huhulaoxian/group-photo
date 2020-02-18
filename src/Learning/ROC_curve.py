
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.metrics import precision_recall_curve, auc,roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import itertools


def classifier_data(y, lower_threshold = 6, upper_threshold = 6):
	for i, score in enumerate(y):
		if y[i] < lower_threshold:
			y[i] = 0
		elif y[i] >= upper_threshold:
			y[i] = 1


def load_data(classify_data,feature):
    X = np.load(feature)	

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
    classify_data = True
    # trainX, trainY, testX, testY = load_data(classify_data)
    X1, y1 = load_data(classify_data,"../../data/feature_vec/f1.npy")
    if classify_data:
        clf = svm.SVC(C = 1, gamma = 2,class_weight='balanced',probability=True)
    else:
        clf = svm.SVR(C = 1, gamma = 3.7)


    k_fold = KFold(n_splits=10, shuffle=True, random_state=12345)
    predictor = clf
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_real, y_proba)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_proba .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(num=1)
    lw = 2
    plt.plot(fpr[0], tpr[0], lw=lw, 
            label='眼睛睁闭特征 (AUC = %0.2f)' % roc_auc[0],marker='o')
            #  label='ROC curve of eye (area = %0.2f)' % roc_auc[0])

######################################################
    X1, y1 = load_data(classify_data,"../../data/feature_vec/f2.npy")

    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    # Compute ROC curve and ROC area for each class
    fpr[0], tpr[0], _ = roc_curve(y_real, y_proba)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_proba .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    plt.plot(fpr[0], tpr[0], lw=lw, 
            label='面部遮挡特征(AUC = %0.2f)' % roc_auc[0],linestyle=':')
            #  label='ROC curve of occlusion(area = %0.2f)' % roc_auc[0],linestyle=':')
################################################
    X1, y1 = load_data(classify_data,"../../data/feature_vec/f3.npy")
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    # Compute ROC curve and ROC area for each class
    fpr[0], tpr[0], _ = roc_curve(y_real, y_proba)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_proba .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2
    plt.plot(fpr[0], tpr[0],lw=lw, 
            label='头部姿态特征(AUC = %0.2f)' % roc_auc[0],marker='*')
            #  label='ROC curve of headpose(area = %0.2f)' % roc_auc[0])
#####################################
    X1, y1 = load_data(classify_data,"../../data/feature_vec/f4.npy")
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    # Compute ROC curve and ROC area for each class
    fpr[0], tpr[0], _ = roc_curve(y_real, y_proba)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_proba .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2
    plt.plot(fpr[0], tpr[0],lw=lw, 
             label='视线特征(AUC = %0.2f)' % roc_auc[0],marker='s')
            #  label='ROC curve of gaze(area = %0.2f)' % roc_auc[0])
#####################################
    X1, y1 = load_data(classify_data,"../../data/feature_vec/f5.npy")
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    # Compute ROC curve and ROC area for each class
    fpr[0], tpr[0], _ = roc_curve(y_real, y_proba)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_proba .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2
    plt.plot(fpr[0], tpr[0],
             lw=lw, label='面部模糊特征(AUC = %0.2f)' % roc_auc[0])
            #  label='ROC curve of blur(area = %0.2f)' % roc_auc[0])
#####################################
    X1, y1 = load_data(classify_data,"../../data/feature_vec/f6.npy")
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    # Compute ROC curve and ROC area for each class
    fpr[0], tpr[0], _ = roc_curve(y_real, y_proba)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_proba .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2
    plt.plot(fpr[0], tpr[0],
             lw=lw, label='微笑特征(AUC = %0.2f)' % roc_auc[0],linestyle='-.')
            #  label='ROC curve of smile(area = %0.2f)' % roc_auc[0],linestyle='-.')
#####################################
    X1, y1 = load_data(classify_data,"../../data/feature_vec/f7.npy")
    y_real = []
    y_proba = []
    for i, (train_index, test_index) in enumerate(k_fold.split(X1)):
        Xtrain, Xtest = X1[train_index], X1[test_index]
        ytrain, ytest = y1[train_index], y1[test_index]
        predictor.fit(Xtrain, ytrain)
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
    
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    # Compute ROC curve and ROC area for each class
    fpr[0], tpr[0], _ = roc_curve(y_real, y_proba)
    roc_auc[0] = auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_real.ravel(), y_proba .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    lw = 2
    plt.plot(fpr[0], tpr[0],
             lw=lw,label='人物位置特征(AUC = %0.2f)' % roc_auc[0],marker='v') 
            #  label='ROC curve of position(area = %0.2f)' % roc_auc[0])

#####################################
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve of Group Feature')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('合影特征的ROC曲线')
    plt.legend(loc="lower right")
    plt.show()