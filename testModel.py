from keras.models import load_model
from keras.utils import np_utils
import os
import pickle

"""
测试模型精度
"""
if __name__ == '__main__':
    model = load_model('./CIFAR10_normal/CIFAR10_model.h5')

    cwd = os.getcwd()
    datafile = cwd + '/CIFAR_10_normal.pkl'  # 标准化后
    # datafile = cwd + '/CIFAR_10_mean.pkl' #零均值化后
    (X_train, y_train), (X_test, y_test) = pickle.load(open(datafile, "rb"))
    print(model.metrics_names)
    y_test = np_utils.to_categorical(y_test, 10 )
    test = model.evaluate(x=X_test,y=y_test,batch_size=128)
    print(test[0]) #loss
    print(test[1]) #acc