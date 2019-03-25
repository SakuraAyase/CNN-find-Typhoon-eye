import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math
from PIL import Image

# 2048x2048.jpg  size: 2048 x 2048
# size128.jpg    size: 512 x 512
# size256.jpg    size: 512 x 512

#
def geo_to_pixel_2048x2048(longitude, latitude):
    if (longitude <= 50.0 and longitude >= -130.0) or longitude > 180.0 or abs(latitude) >= 90.0:
        print("Wrong latitude or longitude inputs!")
        return (0, 0)
    if longitude < -130.0:
        longtitude = 180.0 + abs(longitude + 180.0)
    theta_x = longitude - 50
    theta_y = 90 - latitude
    radias_a = 1024 - 9
    radias_b = radias_a * 1.13

    if theta_x < 90:
        tangent = math.tan(math.radians(theta_x))
        x = radias_a + 9 - math.sqrt(1 / (1 / (radias_a * radias_a) + (tangent * tangent) / (radias_b * radias_b)))
    elif theta_x > 90:
        tangent = math.tan(math.radians(180 - theta_x))
        x = radias_a + 9 + math.sqrt(1 / (1 / (radias_a * radias_a) + (tangent * tangent) / (radias_b * radias_b)))
    else:
        x = radias_a + 9
    if theta_y < 90:
        tangent = math.tan(math.radians(theta_y))
        y = radias_a + 9 - math.sqrt(1 / (1 / (radias_a * radias_a) + (tangent * tangent) / (radias_b * radias_b)))
    elif theta_y > 90:
        tangent = math.tan(math.radians(180 - theta_y))
        y = radias_a + 9 - math.sqrt(1 / (1 / (radias_a * radias_a) + (tangent * tangent) / (radias_b * radias_b)))
    else:
        y = radias_a + 9
    return (int(x), int(y))


def pixel_to_geo_2048x2048():
    pass

def geo_to_pixel_512x512():
    pass

def pixel_to_geo_512x512():
    pass

def on_press(event):
    print("my position:" ,event.button,event.xdata, event.ydata)

if __name__ == '__main__':
    fig = plt.figure()
    img = Image.open('dataset/dataset/imgs/north/200101/4_2048x2048.jpg')
    print('Input Longitude & Latitude:')
    longi = float(input())
    lati = float(input())
    (x, y) = geo_to_pixel_2048x2048(longi, lati)
    print(x,y)
    plt.plot([9, 1024, 2037, 1024, x], [1024, 9, 1024, 2037, y], 'r*')
    plt.imshow(img, animated= True)
    fig.canvas.mpl_connect('button_press_event', on_press)
    plt.show()
