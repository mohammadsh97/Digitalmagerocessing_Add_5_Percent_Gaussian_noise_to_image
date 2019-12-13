#mohammad sharabati: 208979096
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
abs_path = os.getcwd()
image_list = []
img_path = abs_path + '/man.jpg';
img = cv2.imread(img_path, 0)
image_list.append(img)
def Sharpen(img):
    grad = cv2.filter2D(img, -1,np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]]))
    return  grad

def Difference_pictures(bler , hidud):
    im = bler - hidud
    return im

plt.imshow(img, cmap='gray'),plt.title("Original Image "),plt.xticks([]),plt.yticks([])
plt.subplot(221),plt.imshow(img, cmap='gray')
plt.subplot(222),plt.imshow(cv2.blur(img,(3,3)), cmap='gray'), plt.title("blur Image"), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(Sharpen(cv2.blur(img,(3,3))), cmap='gray'), plt.title("Sharpen"), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(Difference_pictures(cv2.blur(img,(3,3)), Sharpen(cv2.blur(img,(3,3)))), cmap='gray'), plt.title("Difference_pictures"), plt.xticks([]), plt.yticks([])
plt.show()