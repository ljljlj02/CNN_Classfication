"""
类VGGNet模型的搭建和训练
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
import keras.backend.tensorflow_backend as KTF
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import callbacks

import os,time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf

num_classes = 10

"""
配置GPU
"""
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

KTF.set_session(sess)
"""
数据加载
"""
cwd = os.getcwd()
datafile = cwd + '/CIFAR_10_normal.pkl' #标准化后
#datafile = cwd + '/CIFAR_10_mean.pkl' #零均值化后
(X_train, y_train), (X_test, y_test) = pickle.load(open(datafile,"rb"))
print('load successfully')
# one hot 编码， 10维列向量
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

# #测试集里误差不在下降时，停止训练
# earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0,patience=10, verbose=1, mode='auto')
#如果验证损失下降，则在每个训练轮后保存模型
checkpointer = ModelCheckpoint(filepath='tmp/bestmodel.h5', monitor='val_loss',verbose=1, save_best_only=True,
                               save_weights_only=False,period=1)
#记录损失历史
class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs= None ):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('acc'))
history = LossHistory()
"""
仿VGG16搭建
"""
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(32, 32,3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_last',kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_last'))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_last',kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_last'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_last'))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_last',kernel_initializer='he_normal'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_last',kernel_initializer='he_normal'))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),data_format='channels_last'))

# model.add(Conv2D(512, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
# model.add(Conv2D(512, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
#
# model.add(Conv2D(512, (3, 3), padding='same', activation='relu',data_format='channels_last',kernel_initializer='he_normal'))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

"""
超参数设定
"""
epochs = 100
lrate = 0.01
batchsize = 128
start = time.time() #记录训练时间
#adam = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,amsgrad=False)
sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
"""
对训练集进行数据增强，适当作
"""
datagen = ImageDataGenerator(
    rotation_range = 20 ,  # 随机旋转的度数范围
    width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip = True,  # 随机水平翻转
)
datagen.fit(X_train) #应用于训练数据
gen = datagen.flow(X_train,y_train,batch_size=batchsize)
hist = model.fit_generator(gen, validation_data=(X_test, y_test),steps_per_epoch=50000//batchsize,
          epochs=epochs, shuffle=True, callbacks=[checkpointer,history])

# 打印测试集上的精度
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#精度、损失图
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['train','test'])
plt.title('loss')
plt.savefig("loss.png",dpi=300,format="png")
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['train','test'])
plt.title('accuracy')
plt.savefig("accuracy.png",dpi=300,format="png")
model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
# model.save_weights("CIFAR10_model_weights.h5")
# model.save('CIFAR10_model.h5')
print("Saved model to disk")
print('Total Time Spent: %.2f seconds' % (time.time() - start))
#输出损失历史到文本文件
lossfile= open('trainacc.txt','a')
for i in range(len(history.losses)):
    s = "Epoch %d  acc:%.4f \n"% (i,history.losses[i])
    lossfile.write(s)
lossfile.close()