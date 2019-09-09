import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/sunflower.jpg')
imageCopy = np.copy(image)

imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

plt.imshow(imageCopy)

gray = cv2.cvtColor(imageCopy, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')

#Implementing Canny Edge Detection

#Define Thresholds
lower = 120
upper = 240

edges = cv2.Canny(gray, lower, upper)
plt.imshow(edges, cmap='gray')

wide = cv2.Canny(gray, 30, 100)
tight = cv2.Canny(gray, 200, 240)

# Display the images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('wide')
ax1.imshow(wide, cmap='gray')

ax2.set_title('tight')
ax2.imshow(tight, cmap='gray')
