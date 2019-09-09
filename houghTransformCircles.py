import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/round_farms.jpg')
imageCopy = np.copy(image)

imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

plt.imshow(imageCopy)

gray = cv2.cvtColor(imageCopy, cv2.COLOR_RGB2GRAY)
gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

# for drawing circles on
circles_im = np.copy(image)

minDist = 45
minRadius = 20
maxRadius = 40
param1 = 70
param2 = 11

circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, minDist, param1, param2, minRadius, maxRadius)

# convert circles into expected type
circles = np.uint16(np.around(circles))
# draw each one
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(circles_im, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(circles_im, (i[0], i[1]), 2, (0, 0, 255), 3)

plt.imshow(circles_im)

f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,15))

ax1.set_title('Original')
ax1.imshow(gray_blur, cmap='gray')

ax2.set_title('With Circles (Hough Transformed)')
ax2.imshow(circles_im, cmap='gray')
