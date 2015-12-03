#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import argparse
import os
import numpy as np
import imghdr
 
parser = argparse.ArgumentParser()
parser.add_argument("source_dir")
parser.add_argument("target_dir")
args = parser.parse_args()

size = 256

for source_imgpath in os.listdir(args.source_dir):
	print source_imgpath
	img = cv2.imread(args.source_dir+"/"+source_imgpath)
	#print imghdr.what(args.source_dir+"/"+source_imgpath)
	#print img.shape
	height, width, depth = img.shape
	new_height = size
	new_width = size

	if height > width:
		new_width = size * width / height
	else:
		new_height = size * height / width

	crop_height_start = ( size - new_height ) / 2
	crop_height_end = crop_height_start + new_height
	crop_width_start = ( size - new_width) / 2
	crop_width_end = crop_width_start + new_width

	resized_img = cv2.resize(img, (new_width, new_height))
	cropped_img = np.zeros((size,size,3),np.uint8)
#	cropped_img.fill(255) #white ver
	cropped_img[crop_height_start:crop_height_end,crop_width_start:crop_width_end] = resized_img
	cv2.imwrite(args.target_dir+"/"+source_imgpath, cropped_img) 
