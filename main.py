from vit_keras import vit, vit_or, utils
import sys
import os
import numpy as np
import cv2
import time
import tensorflow as tf
import math
from BroadLearningSystem import BLStrain, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes, bls_train_input, bls_train_inputenhance

from tensorflow.python.client import device_lib
def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 未微调

image_size = 224
# base_model = vit.vit_t16(image_size=image_size,activation='sigmoid',pretrained=False,include_top=False,pretrained_top=False)
# base_model = vit.vit_s16(image_size=image_size,activation='sigmoid',pretrained=False,include_top=False,pretrained_top=False)
# base_model = vit.vit_b16(image_size=image_size,activation='sigmoid',pretrained=False,include_top=False,pretrained_top=False)
base_model = vit.vit_s16(image_size=image_size,activation='sigmoid', pretrained=True,include_top=False,pretrained_top=False)
# base_model = vit.vit_s32(image_size=image_size,activation='sigmoid', pretrained=True,include_top=False,pretrained_top=False)
# base_model = vit.vit_b32(image_size=image_size,activation='sigmoid',pretrained=True,include_top=False,pretrained_top=False)
# base_model = vit.vit_h14(image_size=image_size,activation='sigmoid',pretrained=False,include_top=False,pretrained_top=False)

# image_size = 384
# base_model = vit.vit_b32(image_size=image_size,activation='sigmoid',pretrained=True,include_top=False,pretrained_top=False)
# base_model = vit.vit_l16(image_size=image_size,activation='sigmoid',pretrained=False,include_top=False,pretrained_top=False)
# base_model = vit.vit_l32(image_size=image_size,activation='sigmoid',pretrained=True,include_top=False,pretrained_top=False)

def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    fileName = "getdatatime_TransBLS-B_CelebA"
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

# LSAFBD, SCUT-FBP5500 and SCUT-FBP
def transformLabel(label_raw):
    if label_raw == 0: ##
        results = [1, 0, 0, 0, 0]
    elif label_raw == 1:
        results = [0, 1, 0, 0, 0]
    elif label_raw == 2:
        results = [0, 0, 1, 0, 0]
    elif label_raw == 3:
        results = [0, 0, 0, 1, 0]
    else:
        results = [0,0, 0, 0, 1]
    return results

# CelebA
def transformLabel_CelebA(label_raw):
    if label_raw == 0: ##
        results = [1, 0]
    else:
        results = [0, 1]
    return results

def getDatanew(filePath, colorType=cv2.IMREAD_COLOR, resize_interpolation=cv2.INTER_LANCZOS4, needGenImg=False):
    tmpData = []
    tmpLabel = []
    with open(filePath) as txtData:
        lines = txtData.readlines()
        for line in lines:
            file, label = line.strip().split()
            ####将string转换为int
            label=int(label)
            # tmpLabel.append(transformLabel(label))
            tmpLabel.append(transformLabel_CelebA(label))
            url = file
            image = utils.read(url, image_size)
            img_formated = vit.preprocess_inputs(image).reshape(1, image_size, image_size, 3)
            img_flat=base_model.predict(img_formated)
            img_flat = img_flat.ravel()
            tmpData.append(img_flat)
    return np.double(tmpData), np.double(tmpLabel)####返回样本和标签


make_print_to_file(path='./results/')
if __name__ == '__main__':
    print("*****************start*******************")
    t1=time.time()
    
    # 获取数据
    # SCUT-FBP_dataset
    # filePath_train = r'/media/shan/F/XieXS/datasets/SCUT-FBP/train.txt'
    # filePath_test = r'/media/shan/F/XieXS/datasets/SCUT-FBP/test.txt'

    # SCUT-FBP5500_dataset
    # filePath_train = r'F:\XieXS\datasets\SCUT-FBP5500\train.txt'
    # filePath_test = r'F:\XieXS\datasets\SCUT-FBP5500\test.txt'

    # LSAFBD_dataset
    # filePath_train = r'F:\XieXS\datasets\LSAFBD\train.txt'
    # filePath_test = r'F:\XieXS\datasets\LSAFBD\test.txt'

    # CelebA_dataset
    filePath_train = r'F:\XieXS\datasets\CelebA\train.txt'
    filePath_test = r'F:\XieXS\datasets\CelebA\test.txt'

    traindata, trainlabel = getDatanew(filePath_train, colorType=cv2.IMREAD_COLOR, needGenImg=False)
    # np.save('./data/X384/CelebA_H14/traindata.npy', traindata)
    # np.save('./data/X384/CelebA_H14/trainlabel.npy', trainlabel)
    testdata, testlabel = getDatanew(filePath_test, colorType=cv2.IMREAD_COLOR,needGenImg=False)
    # np.save('./data/X384/CelebA_H14/testdata.npy', testdata)
    # np.save('./data/X384/CelebA_H14/testlabel.npy', testlabel)
    t3=time.time()
    datatime = t3-t1
    print("datatime:", datatime)
    
    # LSAFBD参数设置
    # N1 = 25  #  # of nodes belong to each window
    # N2 = 72  #  # of windows -------Feature mapping layer
    # N3 = 3088 #  # of enhancement nodes -----Enhance layer
    # s = 0.14  #  shrink coefficient
    # C = 2**-15 # Regularization coefficient

    #  SCUT-FBP5500参数设置
    # N1 = 14
    # N2 = 56
    # N3 = 1130
    # s = 0.31
    # C = 2**-10

    # SCUT-FBP参数设置
    N1 = 14
    N2 = 56
    N3 = 1130
    s = 0.31
    C = 2**-10

    # CelebA参数设置
    # N1 = 30
    # N2 = 56
    # N3 = 3314
    # s = 0.8
    # C = 2**-10

    # small 32
    # N1 = 102
    # N2 = 32
    # N3 = 3620
    # s = 0.77
    # C = 2**-10

    # base 32
    # N1 = 35
    # N2 = 46
    # N3 = 3618
    # s = 0.68
    # C = 2**-10

    # large 32
    # N1 = 49
    # N2 = 38
    # N3 = 2800
    # s = 0.77
    # C = 2**-10


    for i in range(1):
        # print("*****************start*******************")
        print('-------------------TransBLS---------------------------')
        BLStrain(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
    t2=time.time()
    traintime=t2-t1
    print("traintime is:",traintime)
    print("*****************end*******************")
