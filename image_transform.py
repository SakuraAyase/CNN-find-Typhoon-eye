from PIL import Image
import numpy as np

def img_trans(io):
    weight = 256
    height = 256
    image = Image.open('dataset/dataset/imgs/north/200110/1_size256.jpg')
    image = np.array(image.convert("L"))
    print(image.shape, image.dtype)
    """for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = np.int_(np.power((max(image[i, j], 170)-170)*3/255,0.4)*255)"""
    pil_im = Image.fromarray(image)
    out = pil_im.resize((weight,height))
    out.show()
    data = out.getdata()
    data = np.matrix(data, dtype='float')/256
    new_data = np.reshape(data, (weight,height))
    return new_data

img_trans(1)