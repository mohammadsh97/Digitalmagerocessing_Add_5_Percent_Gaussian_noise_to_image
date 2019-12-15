import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = np.zeros((500, 400 , 3), np.uint8)  # check the np.unit8
# img = cv2.rectangle(img, (100, 100), (300, 400), (255, 255, 255), -1)
def noisy(image):
   image = np.array(image)
   GaussNoiseImg = np.random.normal(0, (0.1 ** 0.5) , (500, 400))
   GaussNoiseImg = GaussNoiseImg.reshape(500, 400)
   GaussNoiseImg = np.array(GaussNoiseImg)
   noiseImg = GaussNoiseImg + image
   noiseImg = np.array(noiseImg)
   return noiseImg

img = cv2.imread("mohammad.jpg" , 0)
imageWithNois = noisy(img)
# construct the Sobel x-axis kernel
kernelX = np.array((
   [-1, 0, 1],
   [-2, 0, 2],
   [-1, 0, 1]))

# construct the Sobel y-axis kernel
kernelY = np.array((
   [-1, -2, -1],
   [0, 0, 0],
   [1, 2, 1]))

def conv(image, kernel):
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

filterNewX = conv(np.array(imageWithNois), kernelX)
filterNewY = conv(np.array(imageWithNois), kernelY)
g = np.sqrt(np.power(filterNewX,2) + np.power(filterNewY,2))
g = np.array(g)

plt.subplot(221), plt.imshow(img.astype('uint8'),cmap="gray"), plt.title('Original')
plt.subplot(222), plt.imshow(imageWithNois.astype('uint8'),cmap="gray"), plt.title('Add 5% Gaussian noise')
plt.subplot(223), plt.imshow(g.astype('uint8'),cmap="gray"), plt.title('Rectangle frame')
plt.show()