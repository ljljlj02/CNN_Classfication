"""
使用梯度上升的方法可视化卷积核
 part of codes are copied from
 Deep Learning with Python 5.4
"""
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

KTF.set_session(sess)
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=32):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(500): #迭代更新次数
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

model = load_model('./CIFAR10_normal/CIFAR10_model.h5')
#model.summary() #打印模型参数
#print(model.get_config()) #打印网络结构
for layer in model.layers:
    print(layer.name) #输出每一层网络的名字，方便获取特定层的网络权重

layer_names = ['conv2d_8'] #需要加载的卷积层的名字,取每一层的前64个卷积核
#layer_names = ['conv2d_1','conv2d_2','conv2d_3','conv2d_4'] #需要加载的卷积层的名字,取每一层的前64个卷积核
for layer_name in layer_names:
    filter_index = 0
    size = 32
    margin = 5
    #This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.axis('off')
    plt.savefig('./filters/'+ layer_name+'.png', dpi=300, format="png")
    #plt.show()

