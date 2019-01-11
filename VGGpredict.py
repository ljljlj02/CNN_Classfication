"""
对Image_data下的图片进行分类
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

from keras.models import load_model

if __name__ == '__main__':
    model = load_model('./CIFAR10_normal/CIFAR10_model.h5')
    label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
                  5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
    num_class = len(label_dict)

    cwd = os.getcwd()
    path = cwd + "/Image_data" #预测图片路径
    pathlist = os.listdir(path)
    imgs = []
    imgs_normal = []
    num = 15    #预测图片的大小
    h = num//3+1
    index = 1
    for i,category in enumerate(pathlist):
        for f in os.listdir(path + "/" + category):
            fullpath = os.path.join(path + "/" + category,f)
            img = plt.imread(fullpath)

            img_normal = image.load_img(fullpath,target_size=(32,32))
            img_normal = image.img_to_array(img_normal)
            img_normal = np.expand_dims(img_normal,axis=0)
            img_normal = img_normal.astype('float')
            img_normal = img_normal/255.0

            predict = model.predict_classes(img_normal)
            classed = label_dict[predict[0]]
            plt.subplot(h,3,index)
            plt.imshow(img)
            title = classed
            plt.title(title,fontsize = 'large')
            plt.axis('off')
            index += 1
    plt.savefig('predict.png')
    plt.show()



