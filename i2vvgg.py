import math
import chainer
import chainer.functions as F
import cv2
import cupy as cp
import numpy as np

from chainer import Variable

class i2vVGG(chainer.FunctionSet):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)  # MSRA scaling
        super(i2vVGG, self).__init__(
            conv1_1=F.Convolution2D(3,   64,  3, wscale=w, stride=1, pad=1),
            conv1_2=F.Convolution2D(64,   64,  3, wscale=w, stride=1, pad=1),
            conv2_1=F.Convolution2D(64,   128,  3, wscale=w, stride=1, pad=1),
            conv2_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_1=F.Convolution2D(128,  256,  3, wscale=w, stride=1, pad=1),
            conv3_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv3_3=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_1=F.Convolution2D(256,  512,  3, wscale=w, stride=1, pad=1),
            conv4_2=F.Convolution2D(512,  512,  3, wscale=w, stride=1, pad=1),
            conv4_3=F.Convolution2D(512, 512,  3, wscale=w, stride=1, pad=1),
            conv5_1=F.Convolution2D(512,  512,  3, wscale=w, stride=1, pad=1),
            conv5_2=F.Convolution2D(512,  512,  3, wscale=w, stride=1, pad=1),
            conv5_3=F.Convolution2D(512, 512,  3, wscale=w, stride=1, pad=1),
            conv6_1=F.Convolution2D(512,  1024,  3, wscale=w, stride=1, pad=1),
            conv6_2=F.Convolution2D(1024,  1024,  3, wscale=w, stride=1, pad=1),
            conv6_3=F.Convolution2D(1024, 1000,  3, wscale=w, stride=1, pad=1),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        h = F.relu(self.conv6_3(h))
        h = F.reshape(F.average_pooling_2d(h, 4),(x_data.shape[0],1000))
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def predict(self, x_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv6_1(h))
        h = F.relu(self.conv6_2(h))
        h = F.relu(self.conv6_3(h))
        h = F.reshape(F.average_pooling_2d(h, 4),(x_data.shape[0],1000))
        h = F.softmax(h)
        return h
