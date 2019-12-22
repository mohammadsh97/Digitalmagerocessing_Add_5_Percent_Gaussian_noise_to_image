import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('ex9aimg.png',0)

kernel = np.ones((3,3))

dilation = cv2.dilate(img,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 1)


f = plt.figure()
f.add_subplot(1,2,1)
plt.imshow(np.rot90(img,2),cmap="gray")
f.add_subplot(1,2,2)
plt.imshow(np.rot90(erosion,2),cmap="gray")
plt.show(block=True)