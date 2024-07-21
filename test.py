import numpy as np
import time
import sys
import os
from BroadLearningSystem import BLStrain, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes, bls_train_input, bls_train_inputenhance


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
    fileName = "TranBLS_l16_scut-fbp"
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

#此时类别是5个类
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


make_print_to_file(path='./test_result/scut-fbp')
if __name__ == '__main__':
    
    t1=time.time()
    
    # 获取数据
    # filePath_train = r'/dssg/home/zn_qcb/XieXS/first_paper/data/wy/train.txt'
    # filePath_test = r'/dssg/home/zn_qcb/XieXS/first_paper/data/wy/test.txt'

    # filePath_train = r'/dssg/home/zn_qcb/XieXS/first_paper/data/hg/train.txt'
    # filePath_test = r'/dssg/home/zn_qcb/XieXS/first_paper/data/hg/test.txt'

    # resize_format = (350, 350)####(128, 128)
    # traindata, trainlabel = getDatanew(filePath_train, resize_format, colorType=cv2.IMREAD_COLOR, needGenImg=False)##不需要旋转图片   #####cv2.IMREAD_COLOR   False
    # traindata = kerf(traindata)
    # testdata, testlabel = getDatanew(filePath_test, resize_format, colorType=cv2.IMREAD_COLOR,needGenImg=False) ###不需要旋转图片     #####cv2.IMREAD_GRAYSCALE
    # testdata = kerf(testdata)
    traindata = np.load("/media/shan/F/XieXS/vit-bls/data/SCUT-FBP_l16/traindata.npy",encoding = "bytes")
    trainlabel = np.load("/media/shan/F/XieXS/vit-bls/data/SCUT-FBP_l16/trainlabel.npy",encoding = "bytes")
    testdata = np.load("/media/shan/F/XieXS/vit-bls/data/SCUT-FBP_l16/testdata.npy",encoding = "bytes")
    testlabel = np.load("/media/shan/F/XieXS/vit-bls/data/SCUT-FBP_l16/testlabel.npy",encoding = "bytes")
    #  celeba参数设置
    # tinny
    # N1 = 48
    # N2 = 140
    # N3 = 1336
    # s = 0.74
    # C = 2**-10
    # small
    # N1 = 13
    # N2 = 38
    # N3 = 2294
    # s = 0.22
    # C = 2**-10
    # base
    # N1 = 22
    # N2 = 38
    # N3 = 2876
    # s = 0.13
    # C = 2**-10
    # large
    N1 = 11
    N2 = 44
    N3 = 3254
    s = 0.25
    C = 2**-10

    for i in range(1):
        print("*****************start*******************")
        print('-------------------TransBLS---------------------------')
        BLStrain(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
        # print('-------------------BLS_AddEnhanceNodes------------------------')
        # BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
        # print('-------------------BLS_AddFeatureEnhanceNodes----------------')
        # BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)
        # print('-------------------BLS_INPUT--------------------------')
        # bls_train_input(traindata[0:10000,:],trainlabel[0:10000,:],traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,l,m)
        # print('-------------------bls_train_inputenhance--------------------------')
        # bls_train_inputenhance(traindata[0:10000,:],trainlabel[0:10000,:],traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,l,m,m2)
        print("*****************end*******************")   
    t2=time.time()
    traintime=t2-t1
    print("traintime is:",traintime)