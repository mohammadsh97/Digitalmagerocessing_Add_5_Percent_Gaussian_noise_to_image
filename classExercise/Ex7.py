#mohammad sharabati: 208979096
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def ndarray():
    img = np.zeros(shape=(200, 200))
    img[50:150, 60:140] = 255
    return img


def gaussianFilter(img, i):
    temp = img.copy()
    row, col = temp.shape
    mean = 0
    var = 100
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = temp + gauss * 0.1 * i
    return noisy

def gradient_intensity(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return G, D


def round_angle(angle):
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif 22.5 <= angle < 67.5:
        angle = 45
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 112.5 <= angle < 157.5:
        angle = 135
    return angle


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
            if img[i, j] == weak:  # check if one of the neighbours is strong (=255 by default)
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

images = []
rectangle = ndarray()
for i in range(9):
    img1 = gaussianFilter(rectangle, i)
    img2, D = gradient_intensity(img1)
    img3 = suppression(np.copy(img2), D)
    img4, weak = threshold(np.copy(img3), t=30, T=50)
    img5 = tracking(np.copy(img4), weak, strong=255)
    images.append(img5)
plt.subplot(331), plt.imshow(images[0], cmap='gray'), plt.title("10%"), plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(images[1], cmap='gray'), plt.title("20%"), plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(images[2], cmap='gray'), plt.title("30%"), plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(images[3], cmap='gray'), plt.title("40%"), plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(images[4], cmap='gray'), plt.title("50%"), plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(images[5], cmap='gray'), plt.title("605"), plt.xticks([]), plt.yticks([])
plt.subplot(337), plt.imshow(images[6], cmap='gray'), plt.title("70%"), plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(images[7], cmap='gray'), plt.title("80%"), plt.xticks([]), plt.yticks([])
plt.subplot(339), plt.imshow(images[8], cmap='gray'), plt.title("90%"), plt.xticks([]), plt.yticks([])
plt.show()