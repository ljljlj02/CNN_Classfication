"""
CIFAR-10 训练集有5*10000张，测试集有1*10000张
大小为32x32x3
存储为6个data_batch 在cifar-10-batches-py目录下
"""
import pickle
import os
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import PIL.Image as image

def load_CIFAR_batch(filename):
  """加载单批次图片"""
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
    """加载整个数据集"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)  # 使变成行向量
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
 # X,Y = load_CIFAR_batch('./cifar-10-batches-py/data_batch_1')
 # print(X.shape)  #(10000, 32, 32, 3)
 # print(Y.shape)  #(10000,)
 # image = X[2,:,:,:]
 # plt.imshow(image)
 # plt.show()

 X_train,y_train,X_test,y_test = load_CIFAR10('./cifar-10-batches-py/')
 # print(X_train.shape)      #(50000, 32, 32, 3)
 # print(X_test.shape)       #(10000, 32, 32, 3)

 # """对数据进行零均值化"""
 # train_mean_R = np.mean(X_train[:,:,:,0])
 # train_mean_G = np.mean(X_train[:,:,:,1])
 # train_mean_B = np.mean(X_train[:,:,:,2])
 # print(train_mean_R)
 # print(train_mean_G)
 # print(train_mean_B)
 # mean = './train_mean.mat'
 # #保存训练集的均值
 # sio.savemat(mean,{'mean_R':train_mean_G,'mean_G':train_mean_G,'mean_B':train_mean_B})
 # #保存数据集为pkl格式
 # cwd = os.getcwd()
 # datafile = cwd + '/CIFAR_10_mean.pkl'
 # X_train[:, :, :, 0] -= train_mean_G
 # X_train[:, :, :, 1] -= train_mean_G
 # X_train[:, :, :, 2] -= train_mean_B

 #以归一化来预处理数据
 # cwd = os.getcwd()
 # datafile = cwd + '/CIFAR_10_normal.pkl'
 # X_train /= 255
 # X_test  /= 255
 # pickle.dump([(X_train, y_train), (X_test, y_test)], open(datafile, "wb"))  # 以pkl保存图片数据
 #
 # categories = {
 #     0: 'airplane',
 #     1: 'automobile',
 #     2: 'bird',
 #     3: 'cat',
 #     4: 'deer',
 #     5: 'dog',
 #     6: 'frog',
 #     7: 'horse',
 #     8: 'ship',
 #     9: 'truck'
 # }
 # classFile = './CIFAR10_Categories.pkl' #保存类别名称
 # pickle.dump((categories), open(classFile, "wb"))

 # (X_train, y_train), (X_test, y_test) = pickle.load(open('CIFAR_10_normal.pkl','rb'))
 # print(X_train[0,:,:,:])

