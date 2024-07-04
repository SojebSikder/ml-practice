import cv2

# read the image
img = cv2.imread('images/logo.jpg')


# iterate over the image and update the pixel values to make cross lines
for i, row in enumerate(img):

  # get the pixel values by iterating
    for j, pixel in enumerate(img):
        if (i == j or i+j == img.shape[0]):

            # update the pixel value to black
            img[i][j] = [0, 0, 0]


# shape prints the tuple (height, weight, channels)
print(img.shape)
# image will be a numpy array of the image
print(img)

# display the image
cv2.imshow('image', img)

# wait for a key to close the window
cv2.waitKey(0)
