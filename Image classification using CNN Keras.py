import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

DIRECTORY = "datasets\\images\leaf"
CATEGORIES = ['Strawberry_fresh', 'Strawberry_scrotch']

data = []

for category in CATEGORIES:
    folder = os.path.join(DIRECTORY, category)
    label = CATEGORIES.index(category)

    for img in os.listdir(folder):
        img = os.path.join(folder, img)
        img_array = cv2.imread(img)
        img_array = cv2.resize(img_array, (100, 100))

        data.append([img_array, label])

random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

X = X/255


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(2, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# model.fit(X, y, epochs=15, validation_split=0.1)

# model.save('image-classification.h5')
model = keras.models.load_model('image-classification.h5')
# model.summary()

img_pred = cv2.imread('images/leaf/Strawberry_fresh/0b444634-b557-45f4-a68a-8e9e38cd6683___RS_HL 2184.JPG')
img_pred = cv2.resize(img_pred, (100, 100))
img_pred = np.array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)
img_pred = img_pred/255

prediction = model.predict(img_pred)
prediction = np.argmax(prediction)

print(CATEGORIES[prediction])

plt.imshow(img_pred[0])

plt.show()
