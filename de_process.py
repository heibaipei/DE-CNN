" this code could run well "

################################################
# The code for DE_preprocess
# 2019.7.28
################################################



import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import scipy.io as sio
import numpy as np
import os
import math


def windows(data, size):
    start = 0
    while ((start+size) < data.shape[0]):
        yield int(start), int(start + size)
        start += (size)

def segment_signal_without_transition(data, label, window_size):

    for (start, end) in windows(data, window_size):
        if((len(data[start:end]) == window_size) and (len(set(label))==1)):
            if(start == 0):
                segments = data[start:end]
                labels_one = label[0]
            else:
                segments = np.vstack([segments, data[start:end]])
                labels_one = np.append(labels_one, label[0])
    return segments, labels_one

def get_label(record):
    l = len(record)
    numbers = []
    i = 0
    while i < l:
        num = ''
        symbol = record[i]
        while '0' <= symbol <= '9':  # symbol.isdigit()
            num += symbol
            i += 1
            if i < l:
                symbol = record[i]
            else:
                break
        i += 1
        # if num != '':
        #     numbers.append(int(num))
        if num !='':
            Digit = int(num)
    return Digit

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def read_file(file):
    data = sio.loadmat(file)
    data = data['data']
    print(data.shape)
    return data

def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2*math.pi*math.e*variance)/2

def decompose(data, use_baseline):
    #trial*channel*sample
    # start_index = 384 #3s pre-trial signals  # 这个是deap 数据开始的地方
    print("decompose data.shape", data.shape)

    data = data.reshape(-1, 1000, 62)
    shape = data.shape
    print("shape", shape)
    num = shape[0]
    data = np.transpose(data, (0, 2, 1))    #-1，62，1000
    print("shape", data.shape)

    frequency = 200
    decomposed_data = np.empty([0, 4, 200*5])  # 128*63
    decomposed_de = np.empty([0, 4, 5])
    for trial in range(num): # 有四十个数据段
        temp_data = np.empty([0, 1000])
        temp_de = np.empty([0, 5])
        for channel in range(62):
            trial_signal = data[trial, channel, 0:]
            if use_baseline == 'T':
                pass
            else:
                base_theta_DE = base_alpha_DE = base_beta_DE = base_gmma_DE = 0

            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gmma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)
            de_theta = de_alpha= de_beta = de_gmma= np.zeros(shape=[0], dtype = float)
            for index in range(5):
                de_theta =np.append(de_theta, compute_DE(theta[index*frequency:(index+1)*frequency])-base_theta_DE)
                de_alpha =np.append(de_alpha, compute_DE(alpha[index*frequency:(index+1)*frequency])-base_alpha_DE)
                de_beta =np.append(de_beta, compute_DE(beta[index*frequency:(index+1)*frequency])-base_beta_DE)
                de_gmma =np.append(de_gmma, compute_DE(gmma[index*frequency:(index+1)*frequency])-base_gmma_DE)
            temp_de = np.vstack([temp_de, de_theta])
            temp_de = np.vstack([temp_de, de_alpha])
            temp_de = np.vstack([temp_de, de_beta])
            temp_de = np.vstack([temp_de, de_gmma])
        temp_trial = temp_data.reshape(-1, 4, 1000)
        temp_trial_de = temp_de.reshape(-1, 4, 5)
        decomposed_data = np.vstack([decomposed_data, temp_trial])
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])
    decomposed_data = decomposed_data.reshape(-1, 62, 4, 1000)
    decomposed_de = decomposed_de.reshape(-1, 62, 4, 5)
    return decomposed_data, decomposed_de

def data_1Dto2D(data, Y=10, X=11):
    
#     print("data.shape", data.shape)
#     data_temp = data
#     for i in {7,8,9,10,11}:
#         data_temp[i]= 0

# #     data_tem[:, 25,26,27,28,29] = 0  #### CP
# #     data_tem[:, 23,33,31,39,21] = 0  ####  T
# #     data_tem[:,7,11,23,31,59] = 0   #### A
# #     data_tem[:, 52,54,58,59,60] = 0   #### OP    
    
#     data = data - data_temp
    data_2D = np.zeros([Y, X])
    data_2D[0] = ( 	   	 0, 	   0,  	   	 0, 	   0,    data[0],  data[1],  data[2], 	      0,  	     0, 	   0, 	 	 0)
    data_2D[1] = (	  	 0, 	   0,  	   	 0,         0,   data[3],        0,  data[4],         0, 	   	 0,   	   0, 	 	 0)
    data_2D[2] = (	  	 0, data[5],    data[6],  data[7],   data[8],  data[9], data[10], data[11],  data[12],  data[13], 	 	 0)
    data_2D[3] = (	  	 0, data[14],  data[15],  data[16],  data[17], data[18], data[19], data[20],  data[21],  data[22], 		 0)
    data_2D[4] = (       0, data[23],  data[24],  data[25],  data[26], data[27], data[28], data[29],  data[30],  data[31],       0)
    data_2D[5] = (	  	 0, data[32],  data[33],  data[34],  data[35], data[36], data[37], data[38],  data[39],  data[40], 		 0)
    data_2D[6] = (	  	 0, data[41],  data[42],  data[43],  data[44], data[45], data[46], data[47],  data[48],  data[49], 		 0)
    data_2D[7] = (	  	 0, 	   0,  data[50],  data[51],  data[52], data[53], data[54], data[55],  data[56], 	   0, 		 0)
    data_2D[8] = (	  	 0, 	   0, 	 	 0,   data[57],  data[58], data[59], data[60], data[61], 	   	 0, 	   0, 		 0)
    data_2D[9] = (	  	 0, 	   0, 	 	 0, 	     0, 	    0,        0, 		 0, 	   0, 	   	 0, 	   0, 		 0)
    #return shape:10*11
    return data_2D

def wgn(x, snr):  ## 功率谱密度？
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data. nonzero ()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
    return data_normalized


# A = 1/(np.sqrt(2*math.pi)*sigma)
# data_normalized[data_normalized.nonzero()] = A*np.e**(-((data_normalized[data_normalized.nonzero()] - mean)**2/(2*sigma**2)))*data[data.nonzero()]
# data_normalized[data_normalized.nonzero()] = wgn(data_normalized[data_normalized.nonzero()],6)
# mean = data_normalized[data_normalized.nonzero()].mean()
# sigma = data_normalized[data_normalized. nonzero ()].std()
# data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
# return shape: 9*9

def get_a_student_eegs(student_data_path):
    '''
    @param mat_path:
    @return student_name:
    @return eegs:
    '''
    import scipy.io
    student_name = os.path.basename(student_data_path).replace('.mat', '')
    a_student_data = scipy.io.loadmat(student_data_path)
    eeg_keys = [i for i in a_student_data.keys() if i[0] != '_']
    eegs = []
    for eeg_key in eeg_keys:
        if get_label(eeg_key) in {1, 6, 9, 10, 14}:  #positive emotion
#         if get_label(eeg_key) in {3, 4, 7, 12, 15}:  #negative emotion
#         if get_label(eeg_key) in {2, 5, 8, 11, 13}:  #middle emotion
            eegs.append(a_student_data[eeg_key])  #存储
    return eegs


def get_all_data(dir):
    students_data_dir = './cat/'
    # students_data_dir = 'data3'
    student_data_paths = os.listdir(dir)
    print(student_data_paths)
    # student_data_paths = get_student_data_paths(students_data_dir)
    labels = []
    eg = []
    i = 0
    for student_data_path in student_data_paths:
        num = ''.join([x for x in student_data_path if x.isdigit()])
        temp = int(num)-1
        label = temp * np.ones((5))
        eegs = get_a_student_eegs(student_data_path = os.path.join(students_data_dir, student_data_path))
        labels.append(label)
        eg.append(eegs)
    label_all = []
    for i in range(len(eg)):
        for j in range(len(eg[i])):
            data_f1 = np.array(eg[i][j])
            data_f1 = np.transpose(data_f1)
            window_size = 200*5
            label = labels[i]
            seg_datas, s = segment_signal_without_transition(data_f1, label, window_size)
            if len(label_all) == 0:
                label_all = s
                seg_data_all = seg_datas
            else:
                label_all = np.append(label_all, s)
                seg_data_all = np.vstack([seg_data_all, seg_datas])
    print("seg_data_all.shape", seg_data_all.shape)
    print("len(label_all)", len(label_all))
    return label_all, seg_data_all


def pre_process(data, labels, y_n):
    # data shape  3495 62 1000
    decomposed_data, decomposed_de = decompose(data, y_n)
    # decomposed_data 3495 62 4 5
    print("decomposed_de.shape", decomposed_de.shape)  # 1000, 62 , 4, 5

    data_inter_cnn = np.empty([0, 10, 11])
    decomposed_de = decomposed_de.transpose([0, 3, 2, 1])
    decomposed_de = decomposed_de.reshape(-1, 4, 62)  # 2400*4*32
    samples = decomposed_de.shape[0]
    print( "samples", samples)  # 5000
    bands = decomposed_de.shape[1]
    data_cnn = np.empty([0, 10, 11])
    for sample in range(samples):
        for band in range(bands):
            data_2D_temp = feature_normalize(data_1Dto2D(decomposed_de[sample, band, :]))
#             data_2D_temp = data_1Dto2D(decomposed_de[sample, band, :])
            data_2D_temp = data_2D_temp.reshape(1, 10, 11)
            data_cnn = np.vstack([data_cnn, data_2D_temp])
    data_cnn = data_cnn.reshape(-1, 5, 4, 10, 11)
    print("final data shape:", data_cnn.shape)
    return data_cnn, labels


def get_student_data_paths(students_data_dir='data'):
    '''
    @param students_data_dir:the directory path where keep the student_datas
    @return student_data_paths
    '''
    student_data_paths = os.listdir(students_data_dir)
    return student_data_paths

if __name__ == '__main__':

    dataset_dir = './cat/'
    use_baseline = "F"
    print("processing: ", dataset_dir, "......")
    label1, data1 = get_all_data(dataset_dir)
#     decomposed_data, decomposed_de = decompose(data1, use_baseline)
    data, labels = pre_process(data1, label1, use_baseline)

import pickle
with open('./DE_dataset/cat_200.pkl', "wb") as fp:
        pickle.dump(data, fp, protocol=4)
with open('./DE_dataset/cat_label_200.pkl', "wb") as fp:
        pickle.dump(labels, fp)
print("over")
# for file in os.listdir(dataset_dir):
#     print("processing: ", dataset_dir, "......")
#     file_path = os.path.join(dataset_dir, file)
#     data, labels = pre_process(data, use_baseline)
#     print("final shape:",data.shape)
#     sio.savemat(result_dir+file,{"data":data, "labels":labels,})