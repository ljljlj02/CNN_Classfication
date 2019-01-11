"""
使用Graphviz工具可视化模型结构
需要grahviz、pydot
"""
from keras.utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import np_utils
import os
import pickle
#添加graphviz环境变量
# os.environ["PATH"] += os.pathsep + 'L:/graphviz_2.38/bin/'
if __name__ == '__main__':

    model = load_model('./CIFAR10_normal/CIFAR10_model.h5')
    plot_model(model, to_file='model.png',show_shapes=True)

    modelImage = mpimg.imread('model.png')
    plt.imshow(modelImage)
    plt.axis('off')
    plt.show()