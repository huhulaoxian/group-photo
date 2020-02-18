#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def classifier_data(y, lower_threshold = 6, upper_threshold = 7):
	for i, score in enumerate(y):
		if y[i] <= lower_threshold:
			y[i] = 0
		elif y[i] >= upper_threshold:
			y[i] = 1

def load_data(classify_data):
	# X = np.load('../../data/with_face_feature_vecs.npy')
	# X = np.load('../../data/without_face_feature_vecs.npy')
    X = np.load('../../data/only_face_feature_vecs.npy')
    # X = np.load('../../data/eye_smile_feature_vecs.npy')
    X = preprocessing.scale(X)
    m = X.shape[0]
    n = X.shape[1]
    print(m,n) 
    y = np.load('../../data/scores.npy')
    if classify_data:
        classifier_data(y)

    training_indices = np.random.choice(m, int(0.8*m), replace = False)
    trX = X[training_indices]
    trY = y[training_indices]
	
    test_indices = []
    for index in range(m):
    	if index not in training_indices:
    		test_indices.append(index)
    tsX = X[test_indices]
    tsY = y[test_indices]

    return trX, trY, tsX, tsY 


def save_model(clf):
	# pickle.dump(clf, open( "model_with_face.p", "wb" ))
	# pickle.dump(clf, open( "model_without_face.p", "wb" ))
	# pickle.dump(clf, open( "model_only_face.p", "wb" ))
	pickle.dump(clf, open( "model_smile_eye.p", "wb" ))

if __name__ == '__main__':
    classify_data = False
    trainX, trainY, testX, testY = load_data(classify_data)
    if classify_data:
        clf = svm.SVC(C = 1, gamma = 3.7,class_weight='balanced')
    else:
        clf = svm.SVR(C = 1, gamma = 3.7)

    clf.fit(trainX, trainY)
	# Saving model
    save_model(clf)

	# Reading model
    # clf = pickle.load(open( "model_with_face.p", "rb" ))
    # clf = pickle.load(open( "model_without_face.p", "rb" ))
    # clf = pickle.load(open( "model_only_face.p", "rb" ))
    clf = pickle.load(open( "model_smile_eye.p", "rb" ))
    prediction = clf.predict(testX)
    mse = mean_squared_error(prediction,testY)
    mae = mean_absolute_error(prediction,testY)
    r2 = r2_score(prediction,testY)
    print("mean_squared_error:",mse)
    print("mean_absolute_error:",mae)
    print("r2:",r2)
    # accuracy = np.mean((prediction == testY)) * 100.0
    # print ("\nTest accuracy: %lf%%" % accuracy)
