import cv2
import helpers
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

# Image data directories
image_dir_training = "day_night_images/training/"
image_dir_test = "day_night_images/test/"

#IMAGE_LIST = helpers.load_dataset(image_dir_training)
IMAGE_LIST = helpers.load_dataset(image_dir_test)
"""
# Select an image and its label by list index
image_index = 0
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]

print("Image shape is", selected_image.shape, "and Label is:", selected_label)

#Coverting to a numpy array
a = np.array(IMAGE_LIST)
#Transforms day to 0 and night to 1
le = LabelEncoder()
a[:,1] = le.fit_transform(a[:,1])

image_index = ((np.where(a[:,1] == 1))[0][0])
selected_image = IMAGE_LIST[image_index][0]
selected_label = IMAGE_LIST[image_index][1]

print("Image shape is", selected_image.shape, "and Label is:", selected_label)
"""

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    width = 1100
    height = 600
    dimensions = (width, height)
    # resize image
    standard_im = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    return standard_im


def encode(label):
    numerical_val = 0
    if label == "day":
        numerical_val = 1
    return numerical_val

def standardize(image_list):
    standard_list = []

    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # Create a numerical label
        binary_label = encode(label)

        brightnessValue = avgBrightness(standardized_im)
        estimatedLabel = estimateLabel(image)

        predictionTrue = 0
        if (binary_label == estimatedLabel):
            predictionTrue = 1

        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, binary_label, brightnessValue, estimatedLabel, predictionTrue))

    return standard_list

def avgBrightness(image):
    convertHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    avgBrightness = np.sum(convertHSV[:, :, 2])/(image.shape[0]*image.shape[1])

    return avgBrightness

def estimateLabel(image):
    estimatedLabel = 0
    if (avgBrightness(image) > 103):
        #Predict this as a day image
        estimatedLabel = 1

    return estimatedLabel



# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)
random.shuffle(STANDARDIZED_LIST)

# Display a standardized image and its label

# Select an image by index
image_num = 0
selected_image = STANDARDIZED_LIST[image_num][0]
selected_label = STANDARDIZED_LIST[image_num][1]

# Display image and data about it
plt.imshow(selected_image)
print(avgBrightness(selected_image))
print("Shape: "+str(selected_image.shape))
print("Label [1 = day, 0 = night]: " + str(selected_label))

a = np.array(STANDARDIZED_LIST)
accuracy = np.sum(a[:,[4]])/a.shape[0]
print(accuracy)