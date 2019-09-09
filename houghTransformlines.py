import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./images/phone.jpg')
imageCopy = np.copy(image)

imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)

#plt.imshow(imageCopy)

gray = cv2.cvtColor(imageCopy, cv2.COLOR_RGB2GRAY)

#plt.imshow(gray, cmap='gray')

#Performing Hough Detection

#Detect Edges
#Define Thresholds
lower = 50
upper = 100

edges = cv2.Canny(gray, lower, upper)
#plt.imshow(edges, cmap='gray')

#Find lines using Hough Transform
#rho and theta define the resolution of detection
rho = 1 #1 pixel
theta = np.pi/180 #1 degree

#Threshold specifies the minimum number of hough space intersections it takes to find a line
threshold = 60
min_line_length = 100
#This is the max allowed gap between discontinous line segments for them to be classified as continous lines.
max_line_gap = 5
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

lineImage = np.copy(image)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(lineImage, (x1,y1), (x2,y2), (0,0,255), 3)

#plt.imshow(lineImage)
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15,15))

ax1.set_title('Original')
ax1.imshow(image, cmap='gray')

ax2.set_title('With Edges (Hough Transformed)')
ax2.imshow(lineImage, cmap='gray')
