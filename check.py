#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import cv2
import argparse
import os
import numpy as np

import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time

import numpy as np
from PIL import Image

import math
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
import random
import six
import six.moves.cPickle as pickle
from six.moves import queue

import i2vvgg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import pylab

parser = argparse.ArgumentParser(
    description='Image inspection using chainer')
parser.add_argument('source_dir', help='Path to inspection image file')
parser.add_argument('--model','-m',default='model', help='Path to model file')
parser.add_argument('--mean', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
args = parser.parse_args()

mean_image = pickle.load(open(args.mean, 'rb'))

model = pickle.load(open(args.model,'rb'))

cuda.get_device(0).use()
model.to_gpu()

size = 256
ok=0
ng=0

for source_dirpath in os.listdir(args.source_dir):
    for source_imgpath in os.listdir(args.source_dir+"/"+source_dirpath):
        img = cv2.imread(args.source_dir+"/"+source_dirpath+"/"+source_imgpath)
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
#    cropped_img.fill(255) white ver
        cropped_img[crop_height_start:crop_height_end,crop_width_start:crop_width_end] = resized_img
        top=left=(size-model.insize)/2
        bottom=model.insize+top
        right=model.insize+left
        cropped_img=cropped_img.astype(np.float32).swapaxes(0,2).swapaxes(1,2)
        cropped_img = cropped_img[:, top:bottom, left:right]
        cropped_img -= mean_image[:,top:bottom,left:right]
        cropped_img /= 255
        x = np.ndarray((1, 3, model.insize, model.insize), dtype=np.float32)
        x[0]=cropped_img
        x=cuda.to_gpu(x)
        score = model.predict(x,train=False)
        score=cuda.to_cpu(score.data)

        categories = np.loadtxt("labels.txt", str, delimiter="\t")
        prediction = zip(score[0].tolist(), categories)
        prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
        top_k=1
        ys_pass=0

        for rank, (score, name) in enumerate(prediction[:top_k], start=1):
            print (args.source_dir+"/"+source_dirpath+"/"+source_imgpath+" "+name+" "+source_dirpath)
            if name == source_dirpath:
                ok+=1
            else:
                ng+=1

print ("collect {}/{}".format(str(ok),str(ok+ng)),file=sys.stderr)
print ("not collect {}/{}".format(str(ng),str(ok+ng)),file=sys.stderr)