"""
Code to go through the image dataset and save LUV matrix associated 
For the 40 percent sampled data, saves the matrix at ../../data/LUV_40p.npy
"""

from scipy import misc
from skimage import color
from skimage import data
from PIL import Image
import numpy as np
import os
import csv
import PIL
import logging

import requests
from json import JSONDecoder
import cv2
import uuid

from PIL import Image
import numpy as np
from aip import AipFace
import base64
import time
import logging

import WuFeatures
import MachajdikFeatures

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='compare1_feature.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel('DEBUG')


good_indices = []
bad_indices = []

def basicFeature(path,csv_path):
	LUV = []
	IV = []
	IS = []
	IH = []
	image_sizes = []
	path = path
	reader = csv.reader(open(csv_path,'r'))
	# for row in reader:
	for row in reader:
		print(row[1])
		current_image = path + row[1]
		img = Image.open(current_image)
		img = img.resize((128, 128), Image.ANTIALIAS) 
		img = np.array(img)

		# aspect ratio Feature
		image_sizes.append([img.shape[0], img.shape[1]])
		if(len(img.shape) ==2 ):
			# bad_indices.append(row[1])
			bad_indices.append(row[1])
			# print(row[1])
			continue

		# LUV Feature
		arr = color.rgb2luv(img)
		imgluv = Image.fromarray(arr.astype('uint8')).convert('RGB')
		# imgluv.show()
		LUV.append(arr)
		
		# HSV Feature
		arr = color.rgb2hsv(img)
		IV_Current = arr[:,:,2]
		IV.append(IV_Current)
		IS_Current = arr[:,:,1]
		IS.append(IS_Current)
		IH_Current = arr[:,:,0]
		IH.append(IH_Current)

		good_indices.append(row[1])

	print("可用图像",len(good_indices),"张")
	print("不可用图像",len(bad_indices),"张")
	print("共处理", len(image_sizes),"张图片尺寸")
	return good_indices,LUV,IV,IS,IH,image_sizes

def wuFeature(path,good_indices):
	path = path
	good_indices = good_indices
	feature_vec = []
	for i, img_path in enumerate(good_indices):
		print ("图：",i+1)
		img = cv2.imread(path+img_path)

		resized_img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
		feature_vector = []

		color = WuFeatures.Color()
		texture = WuFeatures.Texture()

		# The effects of saturation and brightness on emotion (e.g., pleasure, arousal and dominance)
		wu_f1_3 = color.f1_3(img)
		for i in wu_f1_3:
		    feature_vector.append(i)

		# Colorfulness
		feature_vector.append(color.f4(img))

		# W3C colors
		wu_f10_25 = color.f10_25(img)
		for i in wu_f10_25:
			feature_vector.append(i)

		# Gray-Level Co-occurance Matrix
		wu_f41_44 = texture.f41_44(img)
		for i in wu_f41_44:
			feature_vector.append(i)

		# Dynamic features(e.g 'len_statics','degree_statics','abs_degree_statics','len_dynamics','degree_dynamics','abs_degree_dynamics')
		dynamics_f45_50 = MachajdikFeatures.dynamics(img)
		for i in dynamics_f45_50:
			feature_vector.append(i)
            
		# Level of Details
		LOD = MachajdikFeatures.LevelOfDetail(img)
		feature_vector.append(LOD)
		feature_vec.append(feature_vector)
	print("共",len(feature_vec),"张图")
	print("共",len(feature_vec[0]),"维特征")
	return feature_vec

if __name__ == "__main__":	
	path = "../../data/compare1/"
	csv_path = '../../data/image_compare1.csv'
	good_indices,LUV,IV,IS,IH,image_sizes = basicFeature(path,csv_path)
	wuFeatures = wuFeature(path,good_indices)
	np.save('../../data/compare1_good_indices.npy',good_indices)
	np.save('../../data/compare1_LUV.npy',LUV)
	np.save('../../data/compare1_IV.npy',IV)
	np.save('../../data/compare1_IS.npy',IS)
	np.save('../../data/compare1_IH.npy',IH)
	np.save('../../data/compare1_image_sizes.npy',image_sizes)
	np.save('../../data/compare1_WuFeature.npy',wuFeatures)