from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.utils import *
from scipy import *
from keras.optimizers import *

import xlrd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

col = 224
row = 224
channel = 3

def str2int(str):
    for i,s in enumerate(str):
        if s == ' ':
            return int(str[0:i])

def img_trans(file_name):
    im = Image.open(file_name)
    width, height = im.size
    im = im.resize((col,row))
    data = np.asarray(im,dtype='float64')


    """
    new_im = Image.fromarray(data)
    new_im.show()
   for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = np.int_(np.power((max(image[i, j], 170)-170)*3/255,0.4)*255)"""
    return data


def get_inform():

    longitude = []
    latitude = []
    img_list = []
    img_longth = []
    path = "dataset/dataset/infos/north/"
    path_img = "dataset/dataset/imgs/north/"

    file_list = os.listdir(path)
    for file in file_list:
        xlsfile = path + file
        book = xlrd.open_workbook(xlsfile)
        sheet0 = book.sheet_by_index(0)
        row4_data = sheet0.col_values(2)[:]
        row5_data = sheet0.col_values(3)[:]
        img_longth.append(len(row4_data)-1)

        for i in range(1,len(row4_data)):
            p = float(row4_data[i])
            w = float(row5_data[i])

            longitude.append(p)
            latitude.append(w)


    file_list = os.listdir(path_img)
    img_l = 0
    for file in file_list:
        print(file)
        xlsfile = path_img + file + "/"
        for i in range(img_longth[img_l]):
            img_list.append(img_trans(xlsfile + str(i+1)+ "_size256.jpg"))
        img_l = img_l + 1

    path = "dataset/dataset/infos/south/"
    path_img = "dataset/dataset/imgs/south/"
    img_longth = []
    file_list = os.listdir(path)
    for file in file_list:
        xlsfile = path + file
        book = xlrd.open_workbook(xlsfile)
        sheet0 = book.sheet_by_index(0)
        row4_data = sheet0.col_values(2)[:]
        row5_data = sheet0.col_values(3)[:]
        img_longth.append(len(row4_data) - 1)

        for i in range(1, len(row4_data)):
            p = float(row4_data[i])
            w = float(row5_data[i])

            longitude.append(p)
            latitude.append(w)
    print(len(longitude))

    position = np.zeros((len(longitude), 2))
    for i in range(len(longitude)):
        position[i, 0] = longitude[i] + 180
        position[i, 1] = latitude[i]


    print(position)
    #position = position / 360

    file_list = os.listdir(path_img)
    img_l = 0
    for file in file_list:
        print(file)
        xlsfile = path_img + file + "/"
        for i in range(img_longth[img_l]):
            img_list.append(img_trans(xlsfile + str(i + 1) + "_size256.jpg"))

        img_l = img_l + 1


    img_list = np.array(img_list)
    lens = len(img_list)
    img_list.shape = (lens,col, row, channel)
    lens = int(np.ceil(lens * 0.8))
    train_img = img_list
    test_img = img_list[lens:]
    train_pos = position
    test_pos = position[lens:]

    print(np.array(img_list).shape)
    print(position.shape)


    return train_img,test_img,train_pos,test_pos


distance = 12756

def model():
    m = Sequential()

    model = VGG16(input_shape=(row, col, channel), weights='imagenet', include_top=False)
    print('Model loaded.')
    for layer in model.layers[:25]:
        layer.trainable = False
    model.summary()

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))

    top_model.add(Dense(16, activation='relu'))
    top_model.add(BatchNormalization())
    top_model.add(Dense(16, activation='relu'))

    top_model.add(Dense(2))
    m.add(model)
    print(len(model.layers))
    # add the model on top of the convolutional base
    m.add(top_model)

    m.summary()
    return m



def train(x,y):

    m=model()
    m.compile(optimizer=rmsprop(lr=0.01),loss = 'mse',metrics=['accuracy'])
    m.fit(x,y,epochs=10)
    return m

def test(y,y_):
    y_d = (y-y_)*(y-y_)

    print(y)
    print(y_)
    return np.average(y_d)

train_img,test_img,train_pos,test_pos = get_inform()

print(train_pos)
m = train(train_img,train_pos)
y = m.predict(test_img)
print(test(y,test_pos))