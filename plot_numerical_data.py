import xlrd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def str2int(str):
    for i,s in enumerate(str):
        if s == ' ':
            return int(str[0:i])

def img_trans(file_name):
    width = 256
    height = 256
    channel = 1
    image = Image.open(file_name)
    image = np.array(image.convert("L"))

    """for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = np.int_(np.power((max(image[i, j], 170)-170)*3/255,0.4)*255)"""
    pil_im = Image.fromarray(image)
    out = pil_im.resize((width,height))
    data = out.getdata()
    data = np.matrix(data, dtype='float')/256
    new_data = np.reshape(data, (width,height,1))
    return new_data

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
        position[i, 0] = longitude[i]
        position[i, 1] = latitude[i]

    print(position)

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
    img_list.shape = (lens,256,256,1)
    lens = np.ceil(len(img_list)*0.9)
    train_img = img_list[:lens]
    test_img = img_list[lens:]
    train_pos = position[:lens]
    test_pos = position[lens:]

    print(np.array(img_list).shape)
    print(position.shape)



    return train_img,test_img,train_pos,test_pos

"""
    path = "dataset/dataset/infos/north/"
    file_list = os.listdir(path)
    for file in file_list:
        xlsfile = path + file
        book = xlrd.open_workbook(xlsfile)
        sheet0 = book.sheet_by_index(0)
        row4_data = sheet0.col_values(4)
        row5_data = sheet0.col_values(5)
        for i in range(1, len(row4_data)):
            p = str2int(row4_data[i])
            w = str2int(row5_data[i])
            if w == 0:
                continue
            centen_p.append(p)
            max_w.append(w)
    print(len(centen_p))
    print(len(max_w))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(max_w, centen_p, 'g+')
    ax.set_xlabel('1-min Winds (knots)')
    ax.set_ylabel('Min, pressure (millibars)')
    plt.show()
"""
