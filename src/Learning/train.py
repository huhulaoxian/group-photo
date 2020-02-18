#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate


def classifier_data(y, lower_threshold = 6.0, upper_threshold = 6.0):
	for i, score in enumerate(y):
		if y[i] < lower_threshold:
			y[i] = 0
		elif y[i] >= upper_threshold:
			y[i] = 1

# def load_data(classify_data):
# 	# X = np.load('../../data/with_face_feature_vecs.npy')
# 	# X = np.load('../../data/without_face_feature_vecs.npy')
# 	# X = np.load('../../data/only_face_feature_vecs.npy')
# 	X = np.load('../../data/eye_smile_feature_vecs.npy')
# 	m = X.shape[0]
# 	n = X.shape[1]
# 	print(m,n) 
# 	y = np.load('../../data/scores.npy')
# 	if classify_data:
# 		classifier_data(y)

# 	training_indices = np.random.choice(m, int(0.8*m), replace = False)
# 	trX = X[training_indices]
# 	trY = y[training_indices]
	
# 	test_indices = []
# 	for index in range(m):
# 		if index not in training_indices:
# 			test_indices.append(index)
# 	tsX = X[test_indices]
# 	tsY = y[test_indices]

# 	return trX, trY, tsX, tsY 

def load_data(classify_data):
	# X = np.load('../../data/with_face_feature_vecs.npy')
	# X = np.load('../../data/without_face_feature_vecs.npy')
	# X = np.load('../../data/only_face_feature_vecs.npy')
	# X = np.load('../../data/filter_with_face_feature_vecs.npy')
	# X = np.load('../../data/filter_without_face_feature_vecs.npy')
	# X = np.load('../../data/smile_feature_vecs.npy')
	# X = np.load('../../data/eye_feature_vecs.npy')
	# X = np.load('../../data/blur_feature_vecs.npy')
	# X = np.load('../../data/filter_eye_smile_feature_vecs.npy')
	# X = np.load('../../data/eye_smile_feature_vecs.npy')
	# X = np.load('../../data/position_feature_vecs.npy')
	# X = np.load('../../data/feature_vec/top15.npy')
	# X = np.load('../../data/feature_vec/top10.npy')
	# X = np.load('../../data/feature_vec/reg_linear_wrapper10.npy')
	# X = np.load('../../data/feature_vec/svr_linear_wrapper10.npy')
	# X = np.load('../../data/feature_vec/svr_linear_without_facewrapper10.npy')
	# X = np.load('../../data/feature_vec/all_feature.npy')
	# X = np.load('../../data/feature_vec/wrapper10.npy')
	# X = np.load('../../data/feature_vec/withoutface_wrapper10.npy')
	# X = np.load('../../data/feature_vec/onlyface_warpper5.npy')
	# X = np.load('../../data/feature_vec/filter_svc_20.npy')
	# X = np.load('../../data/feature_vec/svc_without_20.npy')
	X = np.load('../../data/feature_vec/svr_20.npy')
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


def save_model(clf):
	# pickle.dump(clf, open( "model_with_face.p", "wb" ))
	# pickle.dump(clf, open( "model_without_face.p", "wb" ))
	# pickle.dump(clf, open( "model_only_face.p", "wb" ))
	# pickle.dump(clf, open( "model_smile_eye.p", "wb" ))
	# pickle.dump(clf, open( "model_eye.p", "wb" ))
	# pickle.dump(clf, open( "model_top15.p", "wb" ))
	# pickle.dump(clf, open( "model_top10.p", "wb" ))
	pickle.dump(clf, open( "filter_svc_20.p", "wb" ))

if __name__ == '__main__':
	classify_data = False
	# trainX, trainY, testX, testY = load_data(classify_data)
	trainX, trainY = load_data(classify_data)
	if classify_data:
		clf = svm.SVC(C = 1, gamma = 2,class_weight='balanced')
		scores = cross_val_score(clf,trainX,trainY,cv = 15,scoring="accuracy")
		# scoring = ['accuracy','precision','recall','f1']
		# scores = cross_validate(clf,trainX,trainY,cv = 10,scoring = 'accuracy')
	else:
		clf = svm.SVR(kernel = 'linear',C = 1)
        # trainX = PolynomialFeatures(degree = 2).fit_transform(trainX)
        # clf = LinearRegression()
        # scores = cross_val_score(clf,trainX,trainY,cv = 10,scoring="neg_mean_absolute_error")
		scores = cross_val_score(clf,trainX,trainY,cv = 10,scoring="r2")
	
	print(scores)
	print("平均：",scores.mean())
    # clf.fit(trainX, trainY)
	# # Saving model
	# save_model(clf)

	# Reading model
    # clf = pickle.load(open( "model_with_face.p", "rb" ))
    # clf = pickle.load(open( "model_without_face.p", "rb" ))
    # clf = pickle.load(open( "model_only_face.p", "rb" ))
    # clf = pickle.load(open( "model_smile_eye.p", "rb" ))
    # clf = pickle.load(open( "model_eye.p", "rb" ))
    # clf = pickle.load(open( "model_top15.p", "rb" ))
	#load testset
    # testX = np.load('../../data/test_eye_feature_vecs.npy')
    # testX = np.load('../../data/test_eye_smile_feature_vecs.npy')
    # prediction = clf.predict(testX)
    # print(prediction)
	
    # accuracy = np.mean((prediction == testY)) * 100.0
    # print ("\nTest accuracy: %lf%%" % accuracy)