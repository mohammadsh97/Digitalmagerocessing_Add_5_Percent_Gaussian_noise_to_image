from random import random

import numpy as np
from skimage.feature import canny
import cv2
import matplotlib.pyplot as plt

img = np.zeros((500, 400 , 3), np.uint8)  # check the np.unit8
img = cv2.rectangle(img, (100, 100), (300, 400), (255, 255, 255), -1)


def convFunc(image, kernel):
   (iH, iW) = image.shape[:2]
   (kH, kW) = kernel.shape[:2]
   pad = (kW - 1) // 2
   image = cv2.copyMakeBorder(image, pad, pad, pad, pad,cv2.BORDER_REPLICATE)
   output = np.zeros((iH, iW), dtype="float32")
   for y in np.arange(pad, iH + pad):
      for x in np.arange(pad, iW + pad):
         roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
         k = (roi * kernel).sum()
         output[y - pad, x - pad] = k
   return output

def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(
        np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)): x = x_idxs[i]
    y = y_idxs[i]
    for t_idx in range(num_thetas):
    # Calculate rho. diag_len is added for a positive index
        rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
        accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos



def add_normal_noise(img,perc=10):
    """
    Adds normally distributed noise with a chance of -perc- %
    :param img: original image [numpy.array]
    :param perc: percentage of noise [int]
    :return: new image with noise [numpy.array]
    """
    if perc<0 or perc>100:
        perc=5;
    newimg = []
    for i in range(0,(img.shape[0])):
        row=[]
        for j in range(0,(img.shape[1])):
            # add noise in chance of perc%
            r = random.randint(1,100)
            if r<=perc:
                row.append(np.random.normal(127,45,None))
            else:
                row.append(img[i][j])
        newimg.append(row)
    newimg = np.array(newimg)
    return newimg

filter =  np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])

clnImg = []
for i in range(img.shape[0]):#this loop to get rid of noise
    indx = []
    for j in range(img.shape[1]):
        if img[i][j][0] > 0 and img[i][j][1] > 0 and img[i][j][2] > 0:
            indx.append(1)
        else:
            indx.append(0)
    clnImg.append(indx)
clnImg = np.array(clnImg)

# conF = convFunc(clnImg,filter)
# conF = add_normal_noise(conF)
plt.subplot(221),plt.imshow(clnImg.astype('uint8'),cmap="gray")
plt.show()