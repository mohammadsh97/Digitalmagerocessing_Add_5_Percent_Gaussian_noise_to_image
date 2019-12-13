import cv2
import numpy as np
import matplotlib.pyplot as plt


img = np.zeros((500, 400, 3), np.uint8)  # check the np.unit8
img = cv2.rectangle(img, (100, 100), (300, 400), (255, 255, 255), -1)

# # Sobal
# y = np.array(([2, 1, 0], (1, 0, -1), [0, -1, -2]))
# x = np.array(([-2, -1, 0], (-1, 0, 1), [0, 1, 2]))
#
x = np.array((
   [-1, 0, 1],
   [-2, 0, 2],
   [-1, 0, 1]), dtype="int8")

# construct the Sobel y-axis kernel
y = np.array((
   [-1, -2, -1],
   [0, 0, 0],
   [1, 2, 1]), dtype="int8")


def noisy(image):

    noiseImg = np.random.normal(0,0.1**0.5,image.shape)
    noiseImg = noiseImg.reshape(image.shape)
    noiseImg = noiseImg + image
    return noiseImg

dst1 = cv2.filter2D(noisy(img), -1, x)
dst2 = cv2.filter2D(noisy(img), -1, y)

plt.subplot(121), plt.imshow(noisy(img)), plt.title('Original1')
plt.subplot(122), plt.imshow(dst1+dst2), plt.title('after convolution')
plt.show()
