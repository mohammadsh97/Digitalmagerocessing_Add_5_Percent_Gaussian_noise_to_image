import numpy as np
from scipy.stats import entropy
import cv2
from matplotlib import pyplot as plt

def entropyPhoto(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)



img = cv2.imread('Getty.jpg',0)

kernel = np.ones((25, 25), np.float32) / 625
blur = cv2.filter2D(img, -1, kernel)
gaussian  = cv2.GaussianBlur(img, (33, 33), 0)
median = cv2.medianBlur(img, 33)

images = [img,blur,gaussian,median]
titles =['Original' , "blur", "gauusian", "median"]


for x in range(4):
    plt.subplot(4, 2, x+1), plt.imshow(images[x], 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(str(titles[x])+'\n'+ str(entropyPhoto(images[x])))

plt.show()