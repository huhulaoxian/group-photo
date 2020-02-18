"""
Code to go through the image dataset and save H,S,V matrices associated
"""

from scipy import misc
from PIL import Image
from skimage import color
from skimage import data
import numpy as np
import PIL
import csv

import os

# Change this to the path of a folder containing images with naming convention: "img<num>.png", eg "img442.jpg"
# path = "../../data/GPD/"
path = "../../data/test/"



def main():
	# Load indices for images that were successfully downloaded
	# good_indices = list(np.load('../../data/good_indices.npy'))
	test_indices = np.array([['test1.jpg'],['test2.jpg'],['test3.jpg'],['test4.jpg'],['test5.jpg'],['test6.jpg']])

	IV = []
	IS = []
	IH = []

	# Use this to extract values for a sample 40% of good indices
	# To use all indices, set subset_indices to good_indices

	for image_name in test_indices:
		print (image_name)
		current_image = path + image_name[0]
		img = Image.open(current_image)
		img = img.resize((128, 128), Image.ANTIALIAS) 
		img = np.array(img)
		arr = color.rgb2hsv(img)
		IV_Current = arr[:,:,2]
		IV.append(IV_Current)
		IS_Current = arr[:,:,1]
		IS.append(IS_Current)
		IH_Current = arr[:,:,0]
		IH.append(IH_Current)


	np.save('../../data/test_IV.npy',IV)
	np.save('../../data/test_IS.npy',IS)
	np.save('../../data/test_IH.npy',IH)
	return

if __name__ == "__main__":
	main()