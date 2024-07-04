import tensorflow as tf
import numpy as np

base_dir = "datasets\\images\\flower_photos"

IMAGE_SIZE = 224
BATCH_SIZE = 64


# preprocess image
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_datagen = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

test_datagen = test_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

# create model
cnn = tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(
    filters=64, padding='same', strides=2, kernel_size=3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(
    filters=32, padding='same', strides=2, kernel_size=3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(
    filters=32, padding='same', strides=2, kernel_size=3, activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(4, activation='softmax'))

cnn.compile(optimizer=tf.keras.optimizers.Adam(),
            loss='categorical_crossentropy', metrics=['accuracy'])

cnn.fit(train_datagen, validation_data=test_datagen, epochs=20)


# save model
# cnn.save('flower-classification.h5')
# cnn = tf.keras.models.load_model('flower-classification.h5')
# cnn.summary()

# predict
img = tf.keras.preprocessing.image.load_img(
    'images/flower_photos/daisy/100080576_f52e8ee070_n.jpg', target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = cnn.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    f'This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence.')
print(class_names)
print(score)
print(np.argmax(score))
print(class_names[np.argmax(score)])
print(100 * np.max(score))
print(predictions)
print(np.argmax(predictions))
print(np.max(predictions))
print(np.argmax(predictions[0]))
print(np.max(predictions[0]))
print(np.argmax(predictions[0]))
print(predictions[0])
