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


# path = "../../data/GPD/"
path = "../../data/test/"

# reader = csv.reader(open('../../data/image.csv','r'))
test_indices = np.array([['test1.jpg'],['test2.jpg'],['test3.jpg'],['test4.jpg'],['test5.jpg'],['test6.jpg']])
# good_indices = []
# bad_indices = []
IV = []
IU = []
IL = []
LUV = []


# for row in reader:
for image_name in test_indices:
	# print(row[1])

	# current_image = path + row[1]
	current_image = path + image_name[0]
	img = Image.open(current_image)
	# img = misc.imread(current_image)
	img = img.resize((128, 128), Image.ANTIALIAS) 
	# img.show()
	img = np.array(img)
	if(len(img.shape) ==2 ):
		# bad_indices.append(row[1])
		# print(row[1])
		continue
	arr = color.rgb2luv(img)
	imgluv = Image.fromarray(arr.astype('uint8')).convert('RGB')
	# imgluv.show()
	LUV.append(arr)

	IV_Current = arr[:,:,2]
	# img_v = Image.fromarray(arr[:,:,2].astype('uint8')).convert('RGB')
	# img_v.show()
	IV.append(IV_Current)

	IU_Current = arr[:,:,1]
	# img_u = Image.fromarray(arr[:,:,1].astype('uint8')).convert('RGB')
	# img_u.show()
	IU.append(IU_Current)

	IL_Current = arr[:,:,0]
	# img_l = Image.fromarray(arr[:,:,0].astype('uint8')).convert('RGB')
	# img_l.show()
	IL.append(IL_Current)
	# good_indices.append(row[1])

# print("可用图像",len(good_indices),"张")
# print("不可用图像",len(bad_indices),"张")
# np.save('../../data/good_indices.npy', good_indices)
# np.save('../../data/bad_indices.npy', bad_indices)
# np.save('LUV_V_40p.npy',IV)
# np.save('LUV_U_40p.npy',IU)
# np.save('LUV_L_40p.npy',IL)
np.save('../../data/test_LUV.npy', LUV)
