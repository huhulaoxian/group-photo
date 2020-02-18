#coding=utf-8
from sklearn import svm
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing #标准化数据模块
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go



def classifier_data(y, lower_threshold = 6.0, upper_threshold = 6.0):
	for i, score in enumerate(y):
		if y[i] < lower_threshold:
			y[i] = 0
		elif y[i] >= upper_threshold:
			y[i] = 1

def load_data(classify_data):
	X = np.load('../../data/feature_vec/all_feature_new.npy')
	X = X[:,:7]
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
        # clf = DecisionTreeClassifier()
        clf = RandomForestClassifier(n_estimators=100,random_state=0)
        # clf = svm.SVC(kernel="linear",C = 1, gamma = 2,class_weight='balanced')
    else:
        # clf = svm.SVR(kernel="linear",C = 1, gamma = 2)
        clf = RandomForestRegressor(n_estimators=100,max_depth=5,random_state=0)
        # clf = LinearRegression()
    clf.fit(trainX,trainY)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]
    print(indices)
    np.save('rf_face_ranking.npy',indices)
    feature_name = np.array([
        'eye','occlusion','headpose','gaze','blur','smile','position',
        # 'Brightness',
        # 'Saturation','Hue','Hue_center','Saturation_center','Brightness_center',
        # 'wave_Hue_1','wave_Hue_2','wave_Hue_3','wave_Saturation_1','wave_Saturation_2',
        # 'wave_Saturation_3','wave_Brightness_1','wave_Brightness_2','wave_Brightness_3',
        # 'wave_Hue_sum','wave_Saturation_sum','wave_Brightness_sum','Image Size','Aspect Ratio',
        # 'Number of patches','patch1_H','patch2_H','patch3_H','patch4_H','patch5_H','patch1_S',
        # 'patch2_S','patch3_S','patch4_S','patch5_S','patch1_B','patch2_B','patch3_B','patch4_B',
        # 'patch5_B','patch1_size','patch2_size','patch3_size','patch4_size','patch5_size',
        # 'low_DOF_H','low_DOF_S','low_DOF_B','pleasure','arousal','dominance','colofulness',
        # 'black','silver','gray','white','maroon','red','purple','fuchsia','green','lime','olive',
        # 'yellow','navy','blue','teal','aqua','H_contrast','H_correlation','H_energy',
        # 'H_homogeneity','S_contrast','S_correlation','S_energy','S_homogeneity','B_contrast',
        # 'B_correlation','B_energy','B_homogeneity','len_statics','degree_statics',
        # 'abs_degree_statics','len_dynamics','degree_dynamics','abs_degree_dynamics','Level_of_detail'
    ])
    # feature_name = np.array([
    #     'f1','f2','f3','f4','f5','f6','f7','f8',
    #     'f9','f10','f11','f12','f13',
    #     'f14','f15','f16','f17','f18',
    #     'f19','f20','f21','f22',
    #     'f23','f24','f25','f26','f27',
    #     'f28','f29','f30','f31','f32','f33','f34',
    #     'f35','f36','f37','f38','f39','f40','f41','f42',
    #     'f43','f44','f45','f46','f47','f48',
    #     'f49','f50','f51','f52','f53','f54','f55',
    #     'f56','f57','f58','f59','f60','f61','f62','f63','f64','f65','f66',
    #     'f67','f68','f69','f70','f71','f72','f73','f74',
    #     'f75','f76','f77','f78','f79','f80',
    #     'f81','f82','f83','f84','f85',
    #     'f86','f87','f88','f89','f90'
    # ])

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(trainX.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f]+1, importances[indices[f]]))

    # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(feature_name[indices][:35], importances[indices][:35],
    #        color="r", align="center")
    # plt.xticks(rotation=-45)
    # plt.show()

    # colors = ['lightslategray',] * 35
    # colors[0] = 'crimson'
    # colors[1] = 'crimson'
    # colors[22] = 'crimson'
    # colors[23] = 'crimson'
    # colors[32] = 'crimson'

    fig = go.Figure(go.Bar(
            x=feature_name[indices][:33],
            y=importances[indices][:33],
            orientation='v',
            name='Group features',
            # name='合影特征',
            marker_color='red'),go.Layout(title={
        'text': "The importance of features",
        # 'text': "各特征对于模型的重要性",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        },
             xaxis={
            'title':'Feature',
            # 'title':'特征',
            },yaxis={
            'title':'Importance',
            # 'title':'重要性指标',
                    }))
    fig.update_layout(xaxis_tickangle=45,showlegend=True,
    legend = go.layout.Legend(x=0.7,y=1))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.show()
    # fig.write_image("reg_importance.pdf")
