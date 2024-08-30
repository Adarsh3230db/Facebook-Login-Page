import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread("E:\SAMHARAS\Internship\PXL_20230507_094330215.MP.jpg", cv2.IMREAD_GRAYSCALE)

# Perform 2D DFT
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# Perform DCT
dct = cv2.dct(np.float32(image))
dct_normalized = cv2.normalize(dct, None, 0, 255, cv2.NORM_MINMAX)

# Perform DWHT
dwht = cv2.walsh(image)

# Perform KL Transform
covariance = np.cov(image.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance)
kl_transform = np.dot(image, eigenvectors)

# Plotting the transformed images
plt.subplot(221), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('2D DFT'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(dct_normalized, cmap='gray')
plt.title('DCT'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(dwht, cmap='gray')
plt.title('DWHT'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(kl_transform, cmap='gray')
plt.title('KL Transform'), plt.xticks([]), plt.yticks([])
plt.show()
