import shutil
import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.cluster import KMeans
from tensorflow.python.keras import Sequential
import os


my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
my_new_model.layers[0].trainable = False


def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def extract_vector(path):
    resnet_feature_list = []
    ifold = []
    for im in os.listdir(path):
        ifold.append(im)
        im = cv2.imread(path + "//" + im)
        im = cv2.resize(im, (224, 224))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())

    return ifold, np.array(resnet_feature_list)


def clusty(imdir, targetdir, number_clusters):
    try:
        os.makedirs(targetdir)
    except OSError:
        pass
    for i in range(number_clusters):
        try:
            os.mkdir("output//" + str(i))
        except OSError:
            pass

    f, arr = extract_vector(imdir)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(arr)
    print(kmeans.labels_)
    for i, m in zip(f, kmeans.labels_):
        shutil.copy(imdir + "//" + i, targetdir + "//" + str(m) + "//" + str(i))


clusty("data", "output", 10) ############ Cluster DataBase


