import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read in the image
image = cv2.imread('images/waffle.jpg')

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

plt.imshow(image_copy)
"""
# Convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
gray = np.float32(gray)
plt.imshow(gray, cmap='gray')
"""

# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)

# Dilate the image
erosion = cv2.erode(image_copy, kernel, iterations = 1)
dilation = cv2.dilate(erosion, kernel, iterations = 1)

#Using opening - Erosion -> Dilation
opening = cv2.morphologyEx(image_copy, cv2.MORPH_OPEN, kernel)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,10))
ax1.set_title('Erosion')
ax1.imshow(erosion, cmap='gray')
ax2.set_title('Dilated Image')
ax2.imshow(dilation, cmap='gray')
ax3.set_title('Using Opening')
ax3.imshow(opening, cmap='gray')

#Performing erosion and dilation separately
dilation = cv2.dilate(image_copy, kernel, iterations = 1)
erosion = cv2.erode(dilation, kernel, iterations = 1)

#Using closing - Dilation -> Erosion
closing = cv2.morphologyEx(image_copy, cv2.MORPH_CLOSE, kernel)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,10))
ax1.set_title('Erosion')
ax1.imshow(erosion, cmap='gray')
ax2.set_title('Dilated Image')
ax2.imshow(dilation, cmap='gray')
ax3.set_title('Using Closing')
ax3.imshow(closing, cmap='gray')
