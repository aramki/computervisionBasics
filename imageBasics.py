import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
"""
image = mpimg.imread('./images/waymo_car.jpg')

#Image Dimensions
print("Image Dimensions: ", image.shape)
"""
"""
    Starting with B & W 
"""
"""
#Covert to GrayScale
convGrayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print("Converted Image Dimensions: ", convGrayScale.shape)
plt.imshow(convGrayScale, cmap='gray')

# Print the value at the centre of the image
x = convGrayScale.shape[1]//2
y = convGrayScale.shape[0]//2
print(convGrayScale[y,x])

# Finds the maximum and minimum grayscale values in this image
max_val = np.amax(convGrayScale)
min_val = np.amin(convGrayScale)
print('Max: ', max_val)
print('Min: ', min_val)
"""
"""
    With Colour Images
"""
"""
#image = mpimg.imread('images/wa_state_highway.jpg')
plt.imshow(image)

# Copying RGB Channels into separate arrays
red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('Red channel')
ax1.imshow(red, cmap='gray')
ax2.set_title('Green channel')
ax2.imshow(green, cmap='gray')
ax3.set_title('Blue channel')
ax3.imshow(blue, cmap='gray')
"""

"""
    Creating Blue Screen
"""
"""
pizzaImage = cv2.imread("./images/pizza_bluescreen.jpg")
print("This image is a(n)", type(pizzaImage))
#Please remember that the image dimensions are displayed as Height x Width x Colour Components
print("Image Dimensions", pizzaImage.shape)

#We need to make a copy and convert the image to RGB
pizzaCopy = np.copy(pizzaImage)
pizzaCopy = cv2.cvtColor(pizzaCopy, cv2.COLOR_BGR2RGB)

plt.imshow(pizzaCopy)

#Identifying Colour thresholds for Blue
lowerBlue = np.array([0,0,210])
upperBlue = np.array([70,70,255])

#Creating masks for Blue area
mask = cv2.inRange(pizzaCopy, lowerBlue, upperBlue)
#Visualize the mask - Black area means that the mask isn't effective there
plt.imshow(mask, cmap='gray')

maskedImage = np.copy(pizzaCopy)

maskedImage[mask != 0] = [0, 0, 0]
plt.imshow(maskedImage, cmap='gray')

#Adding the background
backgroundImage = cv2.imread('./images/space_background.jpg')
backgroundImage = cv2.cvtColor(backgroundImage, cv2.COLOR_BGR2RGB)

croppedImage = backgroundImage[0:514, 0:816]
croppedImage[mask == 0] = [0,0,0]

plt.imshow(croppedImage)

completeImage = croppedImage + maskedImage
plt.imshow(completeImage)
"""

"""
    Coding for Green Screen
"""
"""
carImage = cv2.imread("./images/car_green_screen.jpg")
print("This image is a(n)", type(carImage))
#Please remember that the image dimensions are displayed as Height x Width x Colour Components
print("Image Dimensions", carImage.shape)

#We need to make a copy and convert the image to RGB
carCopy = np.copy(carImage)
carCopy = cv2.cvtColor(carCopy, cv2.COLOR_BGR2RGB)
plt.imshow(carCopy)

#Identifying Colour thresholds for Green
lowerGreen = np.array([36, 25, 25])
upperGreen = np.array([70, 255, 255])

#Creating masks for Green area
mask = cv2.inRange(carCopy, lowerGreen, upperGreen)
#Visualize the mask - Black area means that the mask isn't effective there
plt.imshow(mask, cmap='gray')

maskedImage = np.copy(carCopy)
maskedImage[mask != 0] = [0, 0, 0]
plt.imshow(maskedImage, cmap='gray')

#Adding the background
backgroundImage = cv2.imread('./images/space_background.jpg')
backgroundImage = cv2.cvtColor(backgroundImage, cv2.COLOR_BGR2RGB)
plt.imshow(backgroundImage, cmap='gray')

croppedImage = backgroundImage[0:450, 0:660]
croppedImage[mask == 0] = [0,0,0]

plt.imshow(croppedImage)

completeImage = croppedImage + maskedImage
plt.imshow(completeImage)
"""

"""
    Converting to HSV format
"""

image = cv2.imread('images/water_balloons.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# RGB channels
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

ax1.set_title('Red')
ax1.imshow(r, cmap='gray')

ax2.set_title('Green')
ax2.imshow(g, cmap='gray')

ax3.set_title('Blue')
ax3.imshow(b, cmap='gray')

# Convert from RGB to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# HSV channels
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))

ax1.set_title('Hue')
ax1.imshow(h, cmap='gray')

ax2.set_title('Saturation')
ax2.imshow(s, cmap='gray')

ax3.set_title('Value')
ax3.imshow(v, cmap='gray')

# Define our color selection criteria in HSV values
lower_hue = np.array([160,0,0])
upper_hue = np.array([180,255,255])

# Define our color selection criteria in RGB values
lower_pink = np.array([180,0,100])
upper_pink = np.array([255,255,230])

# Define the masked area in RGB space
mask_rgb = cv2.inRange(image, lower_pink, upper_pink)

# mask the image
masked_image = np.copy(image)
masked_image[mask_rgb==0] = [0,0,0]

# Vizualize the mask
plt.imshow(masked_image)

# Now try HSV!

# Define the masked area in HSV space
mask_hsv = cv2.inRange(hsv, lower_hue, upper_hue)

# mask the image
masked_image = np.copy(image)
masked_image[mask_hsv==0] = [0,0,0]

# Vizualize the mask
plt.imshow(masked_image)
