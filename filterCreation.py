import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('./images/brain_MR.jpg')
plt.imshow(image)
imageCopy = np.copy(image)
imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

plt.imshow(imageCopy)

# Convert to Grayscale
# This will be used for detecting changes in image intensity
imageGray = cv2.cvtColor(imageCopy, cv2.COLOR_RGB2GRAY)

# Creating custom filter
# Identifies vertical edges and ignores horizontal edges
# This is a Sobel X filter
vertKernel = np.array([[-2, 0, 2],
                       [-1, 0, 1],
                       [-2, 0, 2]])

# Perform convolution with the input image using filter2D
filteredImage = cv2.filter2D(imageGray, -1, vertKernel)
#plt.imshow(filteredImage, cmap='gray')

# Take out the noise
gaussianBlurImage = cv2.GaussianBlur(imageGray, (3,3), 0)

# Perform convolution with the input image using filter2D
filteredImageGaussian = cv2.filter2D(gaussianBlurImage, -1, vertKernel)
#plt.imshow(filteredImage, cmap='gray')

# Create a Binary Image
retval, binaryImage = cv2.threshold(filteredImageGaussian, 100, 255, cv2.THRESH_BINARY)
#plt.imshow(binaryImage, cmap='gray')
f, (ax2, ax3, ax4, ax5) = plt.subplots(1,4, figsize=(15,15))

ax2.set_title('GrayScale')
ax2.imshow(imageGray, cmap='gray')

ax3.set_title('With only Gaussian filter')
ax3.imshow(gaussianBlurImage, cmap='gray')

ax4.set_title('With Gaussian and Sobel Vertical Filter')
ax4.imshow(filteredImageGaussian, cmap='gray')

ax5.set_title('Binary Image')
ax5.imshow(binaryImage, cmap='gray')
