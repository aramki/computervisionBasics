import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read in the image
image = cv2.imread('images/thumbs_up_down.jpg')

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

#plt.imshow(image_copy)

# Convert to grayscale
gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')

# Create a 5x5 kernel of ones
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

#Convert the white background to black
retval, binary = cv2.threshold(opening, 200, 230, cv2.THRESH_BINARY_INV)
plt.imshow(binary, cmap='gray')

#Find and draw the contours
if (cv2.__version__ > "3.4.3"):
    print(cv2.__version__)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
else:
    retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
anotherCopy = np.copy(image_copy)
all_contours = cv2.drawContours(anotherCopy, contours, -1, (0,255,0), 2)

plt.imshow(all_contours)

#Contour Orientation and Bounding Rectangle

def orientations(contours):
    """
    Orientation
    :param contours: a list of contours
    :return: angles, the orientations of the contours
    """

    # Create an empty list to store the angles in
    # Tip: Use angles.append(value) to add values to this list
    angles = []
    for contour in contours:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        angles.append(angle)

    return angles


# ---------------------------------------------------------- #
# Print out the orientation values
angles = orientations(contours)
print('Angles of each contour (in degrees): ' + str(angles))


## it returns a new, cropped version of the original image
# Function slightly modified from https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts


def left_hand_crop(image, selected_contour):
    """
    Left hand crop
    :param image: the original image
    :param selectec_contour: the contour that will be used for cropping
    :return: cropped_image, the cropped image around the left hand
    """

    x, y, w, h = cv2.boundingRect(selected_contour)
    #box_image = cv2.rectangle(contours_image, (x, y), (x + w, y + h), (200, 0, 200), 2)

    # Make a copy of the image to crop
    cropped_image = np.copy(image)
    cropped_image = image[y: y + h, x: x + w]

    return cropped_image

## Replace this value
selected_contour = sort_contours(contours)
#print(selected_contour[0])

# ---------------------------------------------------------- #
# If you've selected a contour
if (selected_contour is not None):
    # Call the crop function with that contour passed in as a parameter
    cropped_image = left_hand_crop(image, selected_contour[0])
    plt.imshow(cropped_image, cmap='gray')