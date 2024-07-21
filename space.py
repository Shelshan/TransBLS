# from bayes_opt import BayesianOptimization
import numpy as np
import sys
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic
import math
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
    # fileName = "SCUT-FBP_TransBLS-B32"
    # fileName = "SCUT-FBP5500_TransBLS-B16"
    # fileName = "LSAFBD_TransBLS-B16"
    fileName = "CelebA_TransBLS-L16"
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
make_print_to_file(path='./results/L16/')

# load data
# SCUT-FBP
# traindata = np.load(r"./data/X384/SCUT-FBP_B32/traindata.npy", encoding = "bytes")
# trainlabel = np.load(r"./data/X384/SCUT-FBP_B32/trainlabel.npy", encoding = "bytes")
# testdata = np.load(r"./data/X384/SCUT-FBP_B32/testdata.npy", encoding = "bytes")
# testlabel = np.load(r"./data/X384/SCUT-FBP_B32/testlabel.npy", encoding = "bytes")   

# SCUT-FBP5500
# traindata = np.load(r"/media/shan/F/XieXS/vit-bls/data/hg_b16/traindata.npy", encoding = "bytes")
# trainlabel = np.load(r"/media/shan/F/XieXS/vit-bls/data/hg_b16/trainlabel.npy", encoding = "bytes")
# testdata = np.load(r"/media/shan/F/XieXS/vit-bls/data/hg_b16/testdata.npy", encoding = "bytes")
# testlabel = np.load(r"/media/shan/F/XieXS/vit-bls/data/hg_b16/testlabel.npy", encoding = "bytes")   

# LSAFBD
# traindata = np.load(r"./data/X384/LSAFBD_B32/traindata.npy", encoding = "bytes")
# trainlabel = np.load(r"./data/X384/LSAFBD_B32/trainlabel.npy", encoding = "bytes")
# testdata = np.load(r"./data/X384/LSAFBD_B32/testdata.npy", encoding = "bytes")
# testlabel = np.load(r"./data/X384/LSAFBD_B32/testlabel.npy", encoding = "bytes")   

# CelebA
traindata = np.load(r"./data/celeba_l16/traindata.npy", encoding = "bytes")
trainlabel = np.load(r"./data/celeba_l16/trainlabel.npy", encoding = "bytes")
testdata = np.load(r"./data/celeba_l16/testdata.npy", encoding = "bytes")
testlabel = np.load(r"./data/celeba_l16/testlabel.npy", encoding = "bytes")   

C = 2**-10

def BLSopt(argsDict):

    N1 = int(argsDict["N1"])
    N2 = int(argsDict['N2'])
    N3 = int(argsDict["N3"])
    s = float(argsDict["s"])
# #
    testAcc, testPrec, testRecall, testF1_score, testAUC=BLStrain(traindata, trainlabel,testdata, testlabel, s, C, N1, N2, N3)
    print("***************")
    print('N1:',N1)
    print('N2:',N2)
    print('N3:',N3)
    print('s:',s)
    print("testAcc:",testAcc*100)
    print("testPrec:",testPrec*100)
    print("testRecall:",testRecall*100)
    print("testF1_score:",testF1_score*100)
    print("testAUC:",testAUC*100)
    print("***************")    
    return -testAcc

spaceBL = {
    'N1': hp.quniform('N1', 10,50,1),
    'N2': hp.quniform('N2', 30,150,2),
    'N3': hp.quniform('N3', 1000,4000,2),
    's': hp.quniform('s', 0.01,0.8,0.01)
}

# print(hyperopt.pyll.stochastic.sample(spaceBL))
trials = Trials()
best = fmin(BLSopt, space=spaceBL, algo=tpe.suggest, max_evals=2000, trials=trials)
print(hyperopt.pyll.stochastic.sample(spaceBL))
print("****************")
print('best:',best)

########################################################################################

#Incremental learning
# M1 = 2000 # # of adding new patterns
# print('-------------------BLS_INPUT--------------------------')
# bls_train_input(traindata[0:20000,:],trainlabel[0:20000,:],traindata,trainlabel,testdata,testlabel,s,C,N1,N2,N3,L,M1)

# c = 2**-25
# L = 5
# M1 = 20#  # of adding enhance nodes
# M2 = 20  #  # of adding feature mapping nodes
# M3 = 50#  # of adding enhance nodes
# # c = 2**-30
# def BLSopt(argsDict):
#     N1 = int(argsDict["N1"])
#     N2 = int(argsDict['N2'])
#     N3 = int(argsDict["N3"])
#     s = int(argsDict["s"])
#     # c = int(argsDict['c'])
#     # M1 = int(argsDict["M1"])
#     # M2 = int(argsDict['C'])
#     # M3 = int(argsDict["M1"])

# #     test_acc=BLStrain(traindata, trainlabel,testdata, testlabel, s, c, N1, N2, N3)
#     test_acc = BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, c, N1, N2, N3, L, M1, M2, M3)

#     return  test_acc

# spaceBL = {
#     'N1': hp.quniform('N1', 6,15,1),
#     'N2': hp.quniform('N2', 10,120,2),
#     'N3': hp.quniform('N3', 1500,3000,5),
#     's': hp.quniform('s', 0.01,0.8,0.01)
# #     # 'c': hp.quniform('c', 2**-30,2**-20,2**-30)
# # #     'M1': hp.quniform('M1', 20,600,10),
# # #     'M2': hp.quniform('M2', 20,600,10),
# # #     'M3': hp.quniform('M3', 20,600,10),
# }

# print(hyperopt.pyll.stochastic.sample(spaceBL))
# trials = Trials()
# best = fmin(BLSopt, space=spaceBL, algo=tpe.suggest, max_evals=1000, trials=trials)
# print('best:',best)
############通过txt文件进行数据读取，txt存储有图片的路径和类别
# def net_model1(reduction):
#     base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(350, 350, 3))
#     # for layer in base_model.layers[0:-1]:
#     #     layer.trainable = False
#     base_model.layers[0].trainable = False
#     model1 = base_model.output
#     model = GlobalAveragePooling2D()(model1)
#     model = Dense(int(model.shape[-1]) // reduction, use_bias=False,activation=relu)(model)
#     model = Dense(int(model1.shape[-1]), use_bias=False,activation=hard_sigmoid)(model)
#     model = Multiply()([model1,model])
#     out = model
#     model = Dense(1024)(model)
#     model = BatchNormalization()(model)
#     model = Activation('relu')(model)
#     model = Dropout(0.3)(model)
#     model = Dense(5, activation='softmax')(model)
#     model = Model(inputs=base_model.input, outputs= out)
#     # model.compile(loss="categorical_crossentropy",optimizer=SGD(lr = 0.0001,momentum = 0.9),metrics=["accurary"])
#     # model.fit(X_train, batch_size=1, epochs=1)
#     return model