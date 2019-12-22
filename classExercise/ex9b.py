import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('rice.jpg',0)

resImg=np.zeros(img.shape)

for x in range(img.shape[0]):  # the result is gray photo so this loop to make it black and white
    for y in range(img.shape[1]):
        if img[x][y] < 125:
            resImg[x][y] = 0
        else:
            resImg[x][y]=1

kernel = np.ones((3,3),np.uint8)

kernel[0][1]=0
kernel[1][0]=0
kernel[1][2]=0
kernel[2][1]=0

erosion = cv2.erode(resImg,kernel,iterations = 4)
dilation = cv2.dilate(erosion,kernel,iterations = 2)

dilation=resImg-dilation


f = plt.figure()
f.add_subplot(1,2, 1)
plt.imshow(np.rot90(img,2),cmap="gray")
f.add_subplot(1,2, 2)
plt.imshow(np.rot90(dilation,2),cmap="gray")
plt.show(block=True)


