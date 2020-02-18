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

# def load_data(classify_data):
# 	# X = np.load('../../data/with_face_feature_vecs.npy')
# 	# X = np.load('../../data/without_face_feature_vecs.npy')
#     # X = np.load('../../data/only_face_feature_vecs.npy')
# 	# X = np.load('../../data/filter_with_face_feature_vecs.npy')
# 	# X = np.load('../../data/filter_without_face_feature_vecs.npy')
#     # X = np.load('../../data/feature_vec/onlyface_warpper5.npy')
#     # X = np.load('../../data/feature_vec/wrapper10.npy')
#     # X = np.load('../../data/feature_vec/svr_linear_wrapper10.npy')
#     # X = np.load('../../data/feature_vec/svr_linear_without_facewrapper10.npy')
#     # X = np.load('../../data/feature_vec/svc_without_20.npy')
#     # X = np.load('../../data/feature_vec/filter_svc_20.npy')
#     X = np.load('../../data/feature_vec/rf_all_20.npy')
#     # X = np.load('../../data/feature_vec/rf_without_20.npy')
#     # X = np.load('../../data/feature_vec/rf_face_6.npy')
#     # quadratic_featurizer = PolynomialFeatures(degree=1)
#     X = preprocessing.scale(X)
#     #todo 标准化应在下面
#     m = X.shape[0]
#     n = X.shape[1]
#     print(m,n) 
#     y = np.load('../../data/score_decimal_change.npy')


#     training_indices = np.random.choice(m, int(0.8*m), replace = False)
#     # trX = quadratic_featurizer.fit_transform(X[training_indices])
#     trX = X[training_indices]
#     trY = y[training_indices]
	
#     test_indices = []
#     for index in range(m):
#     	if index not in training_indices:
#     		test_indices.append(index)
#     # tsX = quadratic_featurizer.transform(X[test_indices])
#     tsX = X[test_indices]
#     tsY = y[test_indices]

#     return trX, trY,tsX,tsY

def load_data():
    X = np.load('../../data/feature_vec/rf_all_20.npy')	

	# X = preprocessing.scale(X)
    m = X.shape[0]
    n = X.shape[1]
    print(m,n)
    y = np.load('../../data/score_decimal_change.npy')
    
    trX = X
    trY = y
	
    return trX, trY 


def save_model(clf):
	pickle.dump(clf, open( "rf_all_20.p", "wb" ))

if __name__ == '__main__':
    classify_data = False
    
    ######################
    # scores = []
    # for i in range(100):
    #     trainX, trainY, testX, testY = load_data(classify_data)
    #     clf = RandomForestRegressor(n_estimators=130,max_depth=5,random_state=0)
    #     clf.fit(trainX, trainY)
    #     scores.append(clf.score(testX,testY))
    #     print('2 r-squared', clf.score(testX,testY))
    # scores = np.array(scores)
    # print('平均r2',scores.mean())
    # print('最大r2',np.max(scores))
    #######################
    # clf = RandomForestRegressor(n_estimators=130,max_depth=5,random_state=0)
    # trainX, trainY = load_data(classify_data)
    # scores = cross_val_score(clf,trainX,trainY,cv = 10,scoring="r2")
    # print(scores)
    # print("r2：",scores.mean())
    

    clf = RandomForestRegressor(n_estimators=130,max_depth=5,random_state=0)
    trainX, trainY = load_data()
    clf.fit(trainX, trainY)
    save_model(clf)


	# Reading model
    # clf = pickle.load(open( "model_with_face.p", "rb" ))
    # prediction = clf.predict(testX)
    # mse = mean_squared_error(prediction,testY)
    # mae = mean_absolute_error(prediction,testY)
    # r2 = r2_score(prediction,testY)
    # print("mean_squared_error:",mse)
    # print("mean_absolute_error:",mae)
    # print("r2:",r2)

