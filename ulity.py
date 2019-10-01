from scipy.optimize import brentq
from scipy.interpolate import interp1d
import sys
import sklearn
from tensorflow.python.framework import graph_util
from sklearn import metrics
import os
import pandas as pd
import scipy.io as sio
import tensorflow as tf
import numpy as np
import time
import math
from matplotlib import pyplot as plt
from scipy import interpolate


def minus(item):
    return item-1

def pred_approx_val(arr, treshold):
    array_np = np.copy(arr)
    low_val_indices = arr < treshold
    high_val_indices = arr >= treshold
    array_np[low_val_indices] = 0
    array_np[high_val_indices] = 1
    return array_np


def perf_measure(actual, score, treshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    size = len(score)
    for i in range(size):
        predArr = pred_approx_val(score[i], treshold)
        for j in range(len(score[i])):
            if(predArr[j] != actual[i][j] and predArr[j] == 1):
                FP+=1
            if(predArr[j] == actual[i][j] == 1):
                TP+=1
            if(predArr[j] != actual[i][j] and predArr[j] == 0):
                FN+=1
            if(predArr[j] == actual[i][j] == 0):
                TN+=1
    return TP, FP, TN, FN


def calc_far_frr(FP, FN, totalP, totalN):
    FAR = FP/float(totalP) # False Accept Rate in percentage
    FRR = FN/float(totalN) # False Reject Rate in percentage
    return FAR, FRR
def plot_far_frr(far, frr, n):

    axisVal = np.arange(0, 1.0, 0.01)
    n = str(n)
    # PLOT FAR FRR
    plt.figure()
    lw = 2
    plt.plot(far, axisVal, label='False Accept Rate', color='blue', lw=lw)
    plt.plot(axisVal, frr, label='False Reject Rate', color='red', lw=lw)
    plt.xlim([0.0, 1.050])
    plt.ylim([0.0, 1.050])
    plt.xlabel('Treshold')
    plt.ylabel('FAR or FRR')
    plt.title('FAR and FRR')
    plt.legend(loc="lower right")
    plt.savefig('./DE_result/test'+ n +'.jpg')

def plot_far_frr2(far, frr, n):    # PLOT FAR FRR
    ss = 1 - np.array(frr)
    ss = ss.tolist()
    n = str(n)
    plt.figure()
    lw = 2
    plt.plot(far, ss, label='False Accept Rate', color='blue', lw=lw)
    plt.xlim([0.0, 1.250])
    plt.ylim([0.0, 1.250])
    plt.xlabel('far')
    plt.ylabel('1-frr')
    plt.title('FAR and FRR')
    plt.legend(loc="lower right")
    plt.savefig('./DE_result/test2' + n + '.jpg')


def prepare_graph_far_frr(actual, score, totalP, totalN):
    step = 1
    far = []   ######为什么要弄成词典？
    frr = []   ###而不是list？

    for i in range(0, 100, step):
        _, FP, _, FN = perf_measure(actual, score, i/float(100))
        a, b = calc_far_frr(FP, FN, totalP, totalN)
        far.append(a)
        frr.append(b)
    return far, frr

def get_index():
    #
    test_index = []
    for i in range(0,40):
        temp_index = [j for j in range(i*60, i*60+30)]  #一般作为测试集？
        test_index = np.append(test_index, temp_index)

    fine_tune_index = np.setxor1d([i for i in range(0, 2400)], test_index)

    test_index = list(map(int,test_index))
    fine_tune_index = list(map(int,fine_tune_index))
    return test_index, fine_tune_index

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv1d(x, filter_width, in_channels, out_channels, kernel_stride,name):
    weight = weight_variable([filter_width, in_channels, out_channels],name)
    bias = bias_variable([out_channels],name)  # each feature map shares the same weight and bias
    return tf.nn.elu(tf.add(tf.nn.conv1d(x, weight, kernel_stride), bias))

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride,name):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels],name)
    bias = bias_variable([out_channels],name)  # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    return tf.nn.relu(tf.add(conv2d(x, weight, kernel_stride),bias))

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size,name):
    fc_weight = weight_variable([x_size, fc_size],name)
    fc_bias = bias_variable([fc_size],name)
    return tf.nn.relu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size,name):
    readout_weight = weight_variable([x_size, readout_size],name)
    readout_bias = bias_variable([readout_size],name)
    return tf.add(tf.matmul(x, readout_weight), readout_bias)