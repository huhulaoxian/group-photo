#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
import csv
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go

def classifier_data(y, lower_threshold = 6, upper_threshold = 6):
	for i, score in enumerate(y):
		if y[i] <= lower_threshold:
			y[i] = 0
		elif y[i] >= upper_threshold:
			y[i] = 1


def load_data(classify_data,feature):
	X = np.load('../../data/feature_vec/'+feature+'.npy')
	X = preprocessing.scale(X)
	m = X.shape[0]
	n = X.shape[1]
	# print(m,n) 
	y = np.load('../../data/score_decimal_change.npy')
	if classify_data:
		classifier_data(y)
	trX = X
	trY = y
	
	return trX, trY 


if __name__ == '__main__':
	classify_data = True
	# trainX, trainY, testX, testY = load_data(classify_data)
	score=[]
	features = []
	reader = csv.reader(open("../../data/feature_vec.csv"))
	filter_feature = []
	for index,row in enumerate(reader):
		trainX, trainY = load_data(classify_data,row[0])
		if classify_data:
			clf = svm.SVC(C = 1, gamma = 2,class_weight='balanced')
			scores = cross_val_score(clf,trainX,trainY,cv = 10,scoring="accuracy")
		else:
			# reg = svm.SVR(kernel = "linear",C = 1, gamma = 2)
			trainX = PolynomialFeatures(degree = 1).fit_transform(trainX)
			reg = LinearRegression()
			# reg.fit(trainX, trainY)
			scores = -cross_val_score(reg,trainX,trainY,cv = 10,scoring="r2")
			# scores = - cross_val_score(reg,trainX,trainY,cv = 10,scoring="neg_mean_squared_error")	
		# print("平均精确度：",scores.mean())
		if scores.mean()>=0.51:
			score.append(scores.mean())
			features.append(row[0])
			filter_feature.append(index)
		# features.append(row[0][11:-17])
	print(score)
	print(len(filter_feature))
	np.save("../../data/feature_vec/filter_feature.npy",filter_feature)
	if classify_data:
		# plt.bar(features, score)
		# plt.tick_params(labelsize=13)
		# plt.xlabel('Feature', fontsize=15)
		# plt.ylabel('Accuracy', fontsize=15)
		# plt.title('Accuracy of Classifier', fontsize=15)
		# plt.ylim((0.3, 0.8))
		# plt.xlim((-1, len(features)))
		# plt.hlines(0.5,-1,len(features), colors = "r", linestyles = "dashed")
		# plt.show()
		fig = go.Figure(go.Bar(
        x=features,
        y=score,
        orientation='v'),go.Layout(xaxis={
        'title':'Feature'
        },yaxis={
        'title':'Accuracy',
                }))
		fig.update_layout(xaxis_tickangle=45)
		fig.show()
	else:
		plt.bar(features, score)
		plt.xlabel('Feature')
		plt.ylabel('mean squared error')
		plt.show()	