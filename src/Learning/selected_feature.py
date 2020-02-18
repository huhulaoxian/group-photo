"""
Constructing the final feature vectors with selected features from the initial 56
along with RAG cut features.
"""

from __future__ import division
from scipy import misc
import numpy as np
from skimage import color
from skimage import data
import os
import PIL 
from PIL import Image
from pywt import wavedec2
from sklearn.cluster import KMeans
from disjoint_Set import DisjointUnionSets
import logging

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='compare_all_feature.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel('DEBUG')

global IH, IS, IV, path, image_sizes,graph
#face feature
global faces,blurs,smiles,eyes,position,eyegazes,headposes,covers
global LH, HL, HH, S1, S2, S3
global _f7, _f8, _f9, _f10, _f11, _f12, _f13, _f14, _f15

# Parameter K for Kmeans is set here
kmeans_cluster_num = 12

# Some images (b/w) give zero values on S1, S2, S3 - leading to division by zero
def check_zero(epsilon = 50):
	global S1, S2, S3
	if S1 == 0:
		S1 = epsilon
	if S2 == 0:
		S2 = epsilon
	if S3 == 0:
		S3 = epsilon


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f7_f8_f9(i):
	global S1, S2, S3, LH, HL, HH
	HL = LH = HH = [0]*3
	coeffs = wavedec2(IH[i], 'db1', level = 3)
	LL, (HL[2], LH[2], HH[2]), (HL[1], LH[1], HH[1]), (HL[0], LH[0], HH[0]) = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	# print('S1, S2, S3',S1, S2, S3)
	check_zero()


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f10_f11_f12(i):
	global S1, S2, S3, LL, HL, HH, LH
	HL = LH = HH = [0]*3
	coeffs = wavedec2(IS[i], 'db1', level = 3)
	LL, (HL[2], LH[2], HH[2]), (HL[1], LH[1], HH[1]), (HL[0], LH[0], HH[0]) = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	check_zero()


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f13_f14_f15(i):
	global S1, S2, S3, LL, HL, HH, LH
	HL = LH = HH = [0]*3
	coeffs = wavedec2(IV[i], 'db1', level = 3)
	LL, (HL[2], LH[2], HH[2]), (HL[1], LH[1], HH[1]), (HL[0], LH[0], HH[0]) = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	check_zero()

def segments(dic):
	all_lengths = []
	all_patches = []
	for key in dic:
		all_lengths.append(dic[key][0])
		all_patches.append(dic[key][1]) 
	# print (len(all_lengths), len(all_patches))
	all_lengths = np.array(all_lengths)
	all_patches = np.array(all_patches)
	max_5_indices = all_lengths.argsort()[-5:][::-1]	# np.array
	return all_patches[max_5_indices]

# Returns number of islands in a[][] 
def countIslands(a,cluser_num): 
	n = len(a) 
	m = len(a[0]) 
	if cluser_num != 0:
		for j in range(0, n): 
			for k in range(0, m):
				if a[j][k]!= cluser_num:
					a[j][k] = 0
				else :
					a[j][k] = 1
	else:
		for j in range(0, n): 
			for k in range(0, m):
				if a[j][k] == cluser_num:
					a[j][k] = 1
				else:
					a[j][k] = 0
	dus = DisjointUnionSets(n * m) 

	# The following loop checks for its neighbours 
	# and unites the indexes if both are 1. 
	for j in range(0, n): 
		for k in range(0, m): 

			# If cell is 0, nothing to do 
			if a[j][k] == 0: 
				continue


			# Check all 8 neighbours and do a Union 
			# with neighbour's set if neighbour is 
			# also 1 
			if j + 1 < n and a[j + 1][k] == 1: 
				dus.Union(j * (m) + k, 
						(j + 1) * (m) + k) 
			if j - 1 >= 0 and a[j - 1][k] == 1: 
				dus.Union(j * (m) + k, 
						(j - 1) * (m) + k) 
			if k + 1 < m and a[j][k + 1] == 1: 
				dus.Union(j * (m) + k, 
						(j) * (m) + k + 1) 
			if k - 1 >= 0 and a[j][k - 1] == 1: 
				dus.Union(j * (m) + k, 
						(j) * (m) + k - 1) 
			if (j + 1 < n and k + 1 < m and
					a[j + 1][k + 1] == 1): 
				dus.Union(j * (m) + k, (j + 1) *
							(m) + k + 1) 
			if (j + 1 < n and k - 1 >= 0 and
					a[j + 1][k - 1] == 1): 
				dus.Union(j * m + k, (j + 1) *
							(m) + k - 1) 
			if (j - 1 >= 0 and k + 1 < m and
					a[j - 1][k + 1] == 1): 
				dus.Union(j * m + k, (j - 1) *
							m + k + 1) 
			if (j - 1 >= 0 and k - 1 >= 0 and
					a[j - 1][k - 1] == 1): 
				dus.Union(j * m + k, (j - 1) *
							m + k - 1) 

	# Array to note down frequency of each set 
	c = [0] * (n * m) 
	numberOfIslands = 0
	lis = []
	list = []
	for j in range(n): 
		for k in range(m): 
			if a[j][k] == 1: 
				lis.append((j,k))
				x = dus.find(j * m + k) 
				
				# If frequency of set is 0, 
				# increment numberOfIslands 
				if c[x] == 0: 
					numberOfIslands += 1
					c[x] += 1
				else: 
					c[x] += 1

	list.append(numberOfIslands)
	list.append(lis)
	return list 

#face feature

#open_eye proportion
def f1(i):
	if faces[i] ==0:
		eye = 0
		return eye
	if float(eyes[i]) / faces[i]==1:
		eye = 1
		return eye
	else:
		eye = 1-2**-(float(eyes[i]) / faces[i])
		return eye

#cover proportion
def f2(i):
	if baidufaces[i] ==0:
		cover = 0
		return cover
	if covers[i]==0:
		cover = 1
		return cover
	else:
		cover = 1-2**-(1-(float(covers[i]) / baidufaces[i]))
		return cover

#headpose proportion
def f3(i):
	if faces[i] ==0:
		headpose = 0
		return headpose
	else:
		headpose = float(headposes[i]) / faces[i]
		return headpose

#eyegaze proportion
def f4(i):
	if faces[i] ==0:
		eyegaze = 0
		return eyegaze
	if float(eyegazes[i]) / faces[i]==1:
		eyegaze = 1
		return eyegaze
	else:
		eyegaze = 1-2**-(float(eyegazes[i]) / faces[i])
		return eyegaze

#blur proportion
def f5(i):
	if faces[i] ==0:
		return 0
	if blurs[i]==0:
		return 1
	else:
		return 1-2**-(1-(float(blurs[i]) / faces[i]))

#smile proportion
def f6(i):
	if faces[i] ==0:
		smile = 0
		return smile
	else:
		smile = float(smiles[i]) / faces[i]
		return smile

#position
def f7(i):
	if faces[i] ==0:
		return 0
	else:
		if (float(position[i]) > (image_sizes[i][1] / 5 ) * 2) and (float(position[i]) < (image_sizes[i][1] / 5) * 3 ):
			return 1
		else:
			return 0

# Exposure of Light Brightness
def f8(i):
	return sum(sum(IV[i]))/(IV.shape[0] * IV.shape[1])


# Average Saturation / Saturation Indicator
def f9(i):
	return sum(sum(IS[i]))/(IS.shape[0] * IS.shape[1])	


# Average Hue / Hue Indicator
def f10(i):
	return sum(sum(IH[i]))/(IH.shape[0] * IH.shape[1])


# Average hue in inner rectangle for rule of thirds inference
def f11(i):
	X = IH[i].shape[0]
	Y = IH[i].shape[1]
	return sum(sum(IH[i, int(X/3) : int(2*X/3), int(Y/3) : int(2*Y/3)])) * 9 / (X * Y)


# Average saturation in inner rectangle for rule of thirds inference
def f12(i):
	X = IS[i].shape[0]
	Y = IS[i].shape[1]
	return sum(sum(IS[i, int(X/3) : int(2*X/3), int(Y/3) : int(2*Y/3)])) * (9/(X * Y))


# Average V in inner rectangle for rule of thirds inference
def f13(i):
	X = IV[i].shape[0]
	Y = IV[i].shape[1]
	return sum(sum(IV[i, int(X/3) : int(2*X/3), int(Y/3) : int(2*Y/3)])) * (9/(X * Y))


# Spacial Smoothness of first level of Hue property
def f14(i):
	global _f7
	prereq_f7_f8_f9(i)
	_f7 = (1/S1)*(sum(sum(HH[0])) + sum(sum(HL[0])) + sum(sum(LH[0])))
	return _f7
	

# Spacial Smoothness of second level of Hue property
def f15(i):
	global _f8
	prereq_f7_f8_f9(i)
	_f8 = (1/S2)*(sum(sum(HH[1])) + sum(sum(HL[1])) + sum(sum(LH[1])))
	return _f8


# Spacial Smoothness of third level of Hue property
def f16(i):
	global _f9
	prereq_f7_f8_f9(i)
	_f9 = (1/S3)*(sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2])))
	return _f9


# Spacial Smoothness of first level of Saturation property
def f17(i):
	global _f10
	prereq_f10_f11_f12(i)
	_f10 = (1/S1)*(sum(sum(HH[0])) + sum(sum(HL[0])) + sum(sum(LH[0])))
	return _f10

# Spacial Smoothness of second level of Saturation property
def f18(i):
	global _f11
	prereq_f10_f11_f12(i)
	_f11 = (1/S2)*(sum(sum(HH[1])) + sum(sum(HL[1])) + sum(sum(LH[1])))
	return _f11


# Spacial Smoothness of third level of Saturation property
def f19(i):
	global _f12
	prereq_f10_f11_f12(i)
	_f12 = (1/S3)*(sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2])))
	return _f12


# Spacial Smoothness of first level of Intensity property
def f20(i):
	global _f13
	prereq_f13_f14_f15(i)
	_f13 = (1/S1)*(sum(sum(HH[0])) + sum(sum(HL[0])) + sum(sum(LH[0])))
	return _f13


# Spacial Smoothness of second level of Intensity property
def f21(i):
	global _f14
	prereq_f13_f14_f15(i)
	_f14 = (1/S2)*(sum(sum(HH[1])) + sum(sum(HL[1])) + sum(sum(LH[1])))
	return _f14


# Spacial Smoothness of third level of Intensity property
def f22(i):
	global _f15
	prereq_f13_f14_f15(i)
	_f15 = (1/S3)*(sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2])))
	return _f15


# Sum of the average wavelet coefficients over all three frequency levels of Hue property
def f23(i):
	f7(i)
	f8(i)
	f9(i)
	return _f7 + _f8 + _f9


# Sum of the average wavelet coefficients over all three frequency levels of Saturation property
def f24(i):
	f10(i)
	f11(i)
	f12(i)
	return _f10 + _f11 + _f12


# Sum of the average wavelet coefficients over all three frequency levels of Intensity property
def f25(i):
	f13(i)
	f14(i)
	f15(i)
	return _f13 + _f14 + _f15


# Image Size feature
def f26(i):
	return image_sizes[i][0] + image_sizes[i][1]


# Aspect Ratio Feature
def f27(i):
	return image_sizes[i][0] / float(image_sizes[i][1])	

# Number of patches > XY/100 pixels, how many disconnected significantly large regions are present
def f28(i, s):
	count = 0
	for si in s:
		if len(si) >= 164:
			count += 1
	return count

# Average Hue value for patch 1
def f29(i, s):
	si = s[0]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 2
def f30(i, s):
	si = s[1]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 3
def f31(i, s):
	si = s[2]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 4
def f32(i, s):
	si = s[3]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 5
def f33(i, s):
	si = s[4]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 1
def f34(i, s):
	si = s[0]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 2
def f35(i, s):
	si = s[1]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 3
def f36(i, s):
	si = s[2]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 4
def f37(i, s):
	si = s[3]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 5
def f38(i, s):
	si = s[4]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 1
def f39(i, s):
	si = s[0]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 2
def f40(i, s):
	si = s[1]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 3
def f41(i, s):
	si = s[2]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 4
def f42(i, s):
	si = s[3]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 5
def f43(i, s):
	si = s[4]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)

# Measure of largest patch
def f44(i, s):
	si = s[0]
	return len(si)/16384


def f45(i, s):
	si = s[1]
	return len(si)/16384

def f46(i, s):
	si = s[2]
	return len(si)/16384


def f47(i, s):
	si = s[3]
	return len(si)/16384


def f48(i, s):
	si = s[4]
	return len(si)/16384

# DoF feature for Hue property
def f49(i):
	prereq_f7_f8_f9(i)
	v1 = v2 = v3 = 0
	sumv1 = sum(sum(LH[2]))
	if sumv1 > 0:
		v1 = sum(sum(abs(LH[2][4:12,4:12]))) / sumv1
	sumv2 = sum(sum(HL[2]))
	if sumv2 > 0:
		v2 = sum(sum(abs(HL[2][4:12,4:12]))) / sumv2
	sumv3 = sum(sum(HH[2]))
	if sumv3 > 0:
		v3 = sum(sum(abs(HH[2][4:12,4:12]))) / sumv3
	if sumv1 == 0:
		v1 = (v2 + v3)/2
	if sumv2 == 0:
		v2 = (v1 + v3)/2
	if sumv3 == 0:
		v3 = (v1 + v2)/2

	return v1 + v2 + v3


# DoF feature for Saturation property
def f50(i):
	prereq_f10_f11_f12(i)
	v1 = v2 = v3 = 0
	sumv1 = sum(sum(LH[2]))
	if sumv1 > 0:
		v1 = sum(sum(abs(LH[2][4:12,4:12]))) / sumv1
	sumv2 = sum(sum(HL[2]))
	if sumv2 > 0:
		v2 = sum(sum(abs(HL[2][4:12,4:12]))) / sumv2
	sumv3 = sum(sum(HH[2]))
	if sumv3 > 0:
		v3 = sum(sum(abs(HH[2][4:12,4:12]))) / sumv3
	if sumv1 == 0:
		v1 = (v2 + v3)/2
	if sumv2 == 0:
		v2 = (v1 + v3)/2
	if sumv3 == 0:
		v3 = (v1 + v2)/2

	return v1 + v2 + v3


# DoF feature for Intensity property
def f51(i):
	prereq_f13_f14_f15(i)
	v1 = v2 = v3 = 0
	sumv1 = sum(sum(LH[2]))
	if sumv1 > 0:
		v1 = sum(sum(abs(LH[2][4:12,4:12]))) / sumv1
	sumv2 = sum(sum(HL[2]))
	if sumv2 > 0:
		v2 = sum(sum(abs(HL[2][4:12,4:12]))) / sumv2
	sumv3 = sum(sum(HH[2]))
	if sumv3 > 0:
		v3 = sum(sum(abs(HH[2][4:12,4:12]))) / sumv3
	if sumv1 == 0:
		v1 = (v2 + v3)/2
	if sumv2 == 0:
		v2 = (v1 + v3)/2
	if sumv3 == 0:
		v3 = (v1 + v2)/2

	return v1 + v2 + v3

def f52(i):
    return wuFeature[i][0]

def f53(i):
    return wuFeature[i][1]

def f54(i):
    return wuFeature[i][2]

def f55(i):
    return wuFeature[i][3]

def f56(i):
    return wuFeature[i][4]  

def f57(i):
    return wuFeature[i][5]

def f58(i):
    return wuFeature[i][6]

def f59(i):
    return wuFeature[i][7]

def f60(i):
    return wuFeature[i][8]

def f61(i):
    return wuFeature[i][9]

def f62(i):
    return wuFeature[i][10]

def f63(i):
    return wuFeature[i][11]

def f64(i):
    return wuFeature[i][12]

def f65(i):
    return wuFeature[i][13]

def f66(i):
    return wuFeature[i][14]

def f67(i):
    return wuFeature[i][15]

def f68(i):
    return wuFeature[i][16]

def f69(i):
    return wuFeature[i][17]

def f70(i):
    return wuFeature[i][18]

def f71(i):
    return wuFeature[i][19]

def f72(i):
    return wuFeature[i][20]

def f73(i):
    return wuFeature[i][21]

def f74(i):
    return wuFeature[i][22]

def f75(i):
    return wuFeature[i][23]

def f76(i):
    return wuFeature[i][24]

def f77(i):
    return wuFeature[i][25]

def f78(i):
    return wuFeature[i][26]

def f79(i):
    return wuFeature[i][27]

def f80(i):
    return wuFeature[i][28]

def f81(i):
    return wuFeature[i][29]

def f82(i):
    return wuFeature[i][30]

def f83(i):
    return wuFeature[i][31]

def f84(i):
    return wuFeature[i][32]

def f85(i):
    return wuFeature[i][33]

def f86(i):
    return wuFeature[i][34]

def f87(i):
    return wuFeature[i][35]

def f88(i):
    return wuFeature[i][36]

def f89(i):
    return wuFeature[i][37]

def f90(i):
    return wuFeature[i][38]

# path = "../../data/GPD/"
path = "../../data/compare1/"

if __name__ == '__main__':
	good_indices = list(np.load('../../data/compare1_good_indices.npy'))
	image_sizes = list(np.load('../../data/compare1_image_sizes.npy'))
	logger.debug('Loading IHSV...')
	IH = np.load('../../data/compare1_IH.npy')
	IS = np.load('../../data/compare1_IS.npy')
	IV = np.load('../../data/compare1_IV.npy')
	logger.debug('IHSV loaded.')
	logger.debug('Loading LUV...')
	LUV = np.load('../../data/compare1_LUV.npy')
	logger.debug('LUV loaded.')
	logger.debug('loading faces feature...')
	faces = np.load('../../data/compare1_faces.npy')
	blurs = np.load('../../data/compare1_blurs.npy')
	eyes = np.load('../../data/compare1_eyes_new.npy')
	smiles = np.load('../../data/compare1_smiles.npy')
	position = np.load('../../data/compare1_meanpostions.npy')
	eyegazes = np.load('../../data/compare1_eyegazes.npy')
	headposes = np.load('../../data/compare1_headposes.npy')
	# covers = np.load('../../data/covers.npy')
	covers = np.load('../../data/compare1_occlusion.npy')
	baidufaces = np.load('../../data/compare1_faces_baidu.npy')
	logger.debug('loading wuFeature...')
	wuFeature = np.load('../../data/compare1_WuFeature.npy')
	logger.debug("wuFeature共%s维特征",len(wuFeature[0]))
	logger.debug('faces feature loaded')
	feature_vec = []
	# test_indices = np.array([['test1.jpg'],['test2.jpg'],['test3.jpg'],['test4.jpg'],['test5.jpg'],['test6.jpg']])
	for i, index in enumerate(good_indices):
		logger.debug("这是图：%s",i+1)
		feature_vec.append([])
		feature_vec[i].append(f1(i)) 
		feature_vec[i].append(f2(i))
		feature_vec[i].append(f3(i))
		feature_vec[i].append(f4(i))
		feature_vec[i].append(f5(i))
		feature_vec[i].append(f6(i))
		feature_vec[i].append(f7(i))
		feature_vec[i].append(f8(i))
		feature_vec[i].append(f9(i))
		feature_vec[i].append(f10(i))
		feature_vec[i].append(f11(i))
		feature_vec[i].append(f12(i))
		feature_vec[i].append(f13(i))
		feature_vec[i].append(f14(i))
		feature_vec[i].append(f15(i))
		feature_vec[i].append(f16(i))
		feature_vec[i].append(f17(i))
		feature_vec[i].append(f18(i))
		feature_vec[i].append(f19(i))
		feature_vec[i].append(f20(i))
		feature_vec[i].append(f21(i))
		feature_vec[i].append(f22(i))
		feature_vec[i].append(f23(i))
		feature_vec[i].append(f24(i))
		feature_vec[i].append(f25(i))
		feature_vec[i].append(f26(i))
		feature_vec[i].append(f27(i))

		logger.debug('Starting K-Means')
		_LUV = LUV[i].reshape((16384, 3))
		dic ={}
		for cluster_num in range(kmeans_cluster_num):
			kmeans = KMeans(n_clusters=kmeans_cluster_num, random_state=0).fit(_LUV)
			centers = kmeans.cluster_centers_
			graph = kmeans.labels_ 
			graph = graph.reshape((128,128))
			dic[cluster_num] = countIslands(graph,cluster_num)
		s = list(segments(dic))
		H = []
		for k in range(5):
			sumh = 0
			for i1, j1 in s[k]:
				sumh += IH[i][i1][j1]
			H.append(sumh)

		feature_vec[i].append(f28(i,s))
		feature_vec[i].append(f29(i,s))
		feature_vec[i].append(f30(i,s))
		feature_vec[i].append(f31(i,s))
		feature_vec[i].append(f32(i,s))
		feature_vec[i].append(f33(i,s))
		feature_vec[i].append(f34(i,s))
		feature_vec[i].append(f35(i,s))
		feature_vec[i].append(f36(i,s))
		feature_vec[i].append(f37(i,s))
		feature_vec[i].append(f38(i,s))
		feature_vec[i].append(f39(i,s))
		feature_vec[i].append(f40(i,s))
		feature_vec[i].append(f41(i,s))
		feature_vec[i].append(f42(i,s))
		feature_vec[i].append(f43(i,s))
		feature_vec[i].append(f44(i,s))
		feature_vec[i].append(f45(i,s))
		feature_vec[i].append(f46(i,s))
		feature_vec[i].append(f47(i,s))
		feature_vec[i].append(f48(i,s))
		feature_vec[i].append(f49(i))
		feature_vec[i].append(f50(i))
		feature_vec[i].append(f51(i))
		feature_vec[i].append(f52(i))
		feature_vec[i].append(f53(i))
		feature_vec[i].append(f54(i))
		feature_vec[i].append(f55(i))
		feature_vec[i].append(f56(i))
		feature_vec[i].append(f57(i))
		feature_vec[i].append(f58(i))
		feature_vec[i].append(f59(i))
		feature_vec[i].append(f60(i))
		feature_vec[i].append(f61(i))
		feature_vec[i].append(f62(i))
		feature_vec[i].append(f63(i))
		feature_vec[i].append(f64(i))
		feature_vec[i].append(f65(i))
		feature_vec[i].append(f66(i))
		feature_vec[i].append(f67(i))
		feature_vec[i].append(f68(i))
		feature_vec[i].append(f69(i))
		feature_vec[i].append(f70(i))
		feature_vec[i].append(f71(i))
		feature_vec[i].append(f72(i))
		feature_vec[i].append(f73(i))
		feature_vec[i].append(f74(i))
		feature_vec[i].append(f75(i))
		feature_vec[i].append(f76(i))
		feature_vec[i].append(f77(i))
		feature_vec[i].append(f78(i))
		feature_vec[i].append(f79(i))
		feature_vec[i].append(f80(i))
		feature_vec[i].append(f81(i))
		feature_vec[i].append(f82(i))
		feature_vec[i].append(f83(i))
		feature_vec[i].append(f84(i))
		feature_vec[i].append(f85(i))
		feature_vec[i].append(f86(i))
		feature_vec[i].append(f87(i))
		feature_vec[i].append(f88(i))
		feature_vec[i].append(f89(i))
		feature_vec[i].append(f90(i))

logger.debug("lenth of features：%s,number of images: %d",len(feature_vec[1]),len(feature_vec))
np.save('../../data/feature_vec/compare1_all_feature.npy', feature_vec)