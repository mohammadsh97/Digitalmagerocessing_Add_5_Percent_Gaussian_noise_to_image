from random import random

import numpy as np
from skimage.feature import canny
import cv2
import matplotlib.pyplot as plt

img = np.zeros((500, 400 , 3), np.uint8)  # check the np.unit8
img = cv2.rectangle(img, (100, 100), (300, 400), (255, 255, 255), -1)

def noisy(image):
   image = np.array(image)
   GaussNoiseImg = np.random.normal(0, (0.1 ** 0.5) , (500, 400))
   GaussNoiseImg = GaussNoiseImg.reshape(500, 400)
   GaussNoiseImg = np.array(GaussNoiseImg)
   noiseImg = GaussNoiseImg + image
   noiseImg = np.array(noiseImg)
   return noiseImg

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

# filter =  np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])

img = np.array(img)
img = img[:,:,0]
noiseImg = noisy(img)
# conF = convFunc(img,filter)
plt.subplot(221),plt.imshow(img.astype('uint8'),cmap="gray")
plt.subplot(222),plt.imshow(noiseImg.astype('uint8'),cmap="gray")
# plt.subplot(223),plt.imshow(conF.astype('uint8'),cmap="gray")
plt.show()