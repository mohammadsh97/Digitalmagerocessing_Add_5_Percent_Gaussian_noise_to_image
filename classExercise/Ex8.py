import cv2
import numpy as np
import matplotlib.pyplot as plt



def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len=int(np.ceil(np.sqrt(width*width+height*height))) #max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos



# Create binary image and call hough_line
image = np.zeros((50,50))
image[10:40, 10:40] = np.eye(30)
res = image[::-1]
finalImage = image + res
accumulator, thetas, rhos = hough_line(finalImage)
plt.subplot(121), plt.imshow(finalImage.astype('uint8'),cmap="gray"), plt.title('Original')
plt.subplot(122), plt.imshow(accumulator.astype('uint8'),cmap="gray"), plt.title('Xْ')
plt.show()




# # Create binary image and call hough_line
# image = np.zeros((50,50))
# image[10:40, 10:40] = np.eye(30)
# accumulator, thetas, rhos = hough_line(image)
# # Easiest peak finding based on max votes
# idx = np.argmax(accumulator)
# rho = rhos[idx // accumulator.shape[1]]
# theta = thetas[idx % accumulator.shape[1]] print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))