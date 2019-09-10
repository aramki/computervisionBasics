import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read in the image
image = cv2.imread('images/monarch.jpg')

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

pixelVals = image_copy.reshape((-1, 3))
pixelVals = np.float32(pixelVals)

#Implementing KMeans
# define stopping criteria
# you can change the number of max iterations for faster convergence!
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k = 12
retval, labels, centers = cv2.kmeans(pixelVals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
labels_reshape = labels.reshape(image.shape[0], image.shape[1])

#plt.imshow(segmented_image)

## Visualizing 1 segment
plt.imshow(labels_reshape==0, cmap='gray')

# mask an image segment by cluster

cluster = 0
masked_image = np.copy(image)
# turn the mask green!
masked_image[labels_reshape == cluster] = [0, 0, 255]

#plt.imshow(masked_image)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
ax1.set_title('Segmented')
ax1.imshow(segmented_image, cmap='gray')
ax2.set_title('Masked')
ax2.imshow(masked_image, cmap='gray')