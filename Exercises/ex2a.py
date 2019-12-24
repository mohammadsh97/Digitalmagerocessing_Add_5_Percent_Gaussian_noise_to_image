from matplotlib import pyplot as plt
import cv2
import numpy as np
import matplotlib.lines as ml

from scipy.ndimage import gaussian_filter


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


def gs_filter(img):
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        flt = [[1, 2, 1],[2, 4, 2], [1, 2, 1]]
        return convFunc(img, np.array(flt))

def gradient_intensity(img):
    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)

    # Apply kernels to the image
    Ix = convFunc(img, Kx)
    Iy = convFunc(img, Ky)

    # return the hypothenuse of (Ix, Iy)
    G = np.hypot(Ix, Ix)  # sqrt(x1**2 + x2**2)
    D = np.arctan2(Iy, Ix)
    return (G, D)

def round_angle(angle):
    if angle < 22.5:
        return 0
    elif angle < 67.5:
        return 45
    elif angle < 113.5:
        return 90
    else:
        return 135

def suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i, j] = img[i, j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i, j] = img[i, j]

            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                pass
    return Z


def threshold(img, t, T):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(50),
        'STRONG': np.int32(255),
    }

    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)

    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)

    return (img, cf.get('WEAK'))


def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                            or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                            or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def imgCopy(img):
    tmp = []
    for x in range(img.shape[0]):
         indx = []
         for y in range(img.shape[1]):
            indx.append(img[x, y])
         tmp.append(indx)
    return np.array(tmp)

def toDrawLine(first, sec): #to draw lines
    getX = plt.gca()
    xmin, xmax = getX.get_xbound()

    if (sec[0] == first[0]):
        xmin = xmax = first[0]
        ymin, ymax = getX.get_ybound()
    else:
        ymax = first[1] + (sec[1] - first[1]) / (sec[0] - first[0]) * (xmax - first[0])
        ymin = first[1] + (sec[1] - first[1]) / (sec[0] - first[0]) * (xmin - first[0])

    l = ml.Line2D([xmin, xmax], [ymin, ymax])
    l.set_color("red")
    getX.add_line(l)
    return l

img = cv2.imread("suduko.jpg", 0)
clnImg = []
for i in range(img.shape[0]):#this loop to get rid of noise
    indx = []
    for j in range(img.shape[1]):
        if img[i][j] > 50:
            indx.append(1)
        else:
            indx.append(0)
    clnImg.append(indx)
clnImg = np.array(clnImg)
img1 = gs_filter(clnImg)
img2, D = gradient_intensity(img1)
img3 = suppression(imgCopy(img2), D)
img4, weak = threshold(imgCopy(img3), 30, 50)
img5 = tracking(imgCopy(img4), weak)
#to find max and min points of the lines
xMax = []
xMin = []
yMax = 67
yMin = 372
for x in range(img5.shape[1]):
    if img5[yMax][x] == 255:
        xMax.append(x)

for x in range(img5.shape[1]):
    if img5[yMin][x] == 255:
        xMin.append(x)

maxLine1 = []
minLine1 = []
maxLine2 = []
minLine2 = []
maxLine3 = []
minLine3 = []
maxLine4 = []
minLine4 = []

idx = 0
for k in range(len(xMax)):
    while idx < len(xMin):
        if abs(xMax[k] - xMin[idx]) <= 25:
            if len(maxLine3) > 0 and len(maxLine4) == 0 and abs(maxLine3[0] - xMax[k]) > 30:
                maxLine4 = [xMax[k], yMax]
                minLine4 = [xMin[idx], yMin]
                break
            if len(maxLine1) == 0:
                maxLine1 = [xMax[k], yMax]
                minLine1 = [xMin[idx], yMin]
                break
            if len(maxLine2) == 0 and abs(maxLine1[0] - xMax[k]) > 30:
                maxLine2 = [xMax[k], yMax]
                minLine2 = [xMin[idx], yMin]
                break
            if len(maxLine2) > 0 and len(maxLine3) == 0 and abs(maxLine2[0] - xMax[k]) > 30:
                maxLine3 = [xMax[k], yMax]
                minLine3 = [xMin[idx], yMin]
                break

                break
        idx = idx + 1
    if idx == len(xMin):
        idx = 0
#final result
plt.imshow(img, cmap="gray")
toDrawLine(maxLine1, minLine1)
toDrawLine(maxLine2, minLine2)
toDrawLine(maxLine3, minLine3)
toDrawLine(maxLine4, minLine4)
plt.show()