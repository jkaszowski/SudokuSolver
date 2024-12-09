import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


train_data = keras.utils.image_dataset_from_directory(
    'images',
    validation_split=0.1,
    seed=1,
    labels='inferred',
    subset="training",
    color_mode='grayscale',
    image_size=(28, 28))

validation_data = keras.utils.image_dataset_from_directory(
    'images',
    validation_split=0.1,
    seed=1,
    labels='inferred',
    subset="validation",
    color_mode='grayscale',
    image_size=(28, 28))


# print(train_data.class_names)
# print(train_data)

# model = keras.Sequential([
#     layers.Input(shape=(28, 28, 1)),  # Adjust input size to MNIST
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(128, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(11, activation='softmax')  # 10 classes for MNIST + background
# ])

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(11, activation="softmax"),
    ]
)

# model = keras.Sequential([
#     # Input layer
#     layers.Input(shape=(28, 28, 1)),
#
#     # Convolutional layer 1
#     layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#
#     # Convolutional layer 2
#     layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#
#     # Convolutional layer 3
#     layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#
#     # Flatten the feature maps
#     layers.Flatten(),
#
#     # Fully connected dense layers
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),  # Dropout for regularization
#     layers.Dense(64, activation='relu'),
#
#     # Output layer with 11 classes
#     layers.Dense(11, activation='softmax')
# ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use sparse or categorical crossentropy
    metrics=['accuracy']
)

history = model.fit(train_data,epochs = 10, validation_data = validation_data)
model.save("model.keras")

plt.clf()
fig, ax1 = plt.subplots()
plt.plot(history.history['loss'],label="loss")
plt.plot(history.history['val_loss'],label="val loss")
# plt.legend(['loss','accuracy','val_loss','val_accuracy'])
ax2 = ax1.twinx()
ax2.set_ylim(0,1)
plt.plot(history.history['val_accuracy'],label="val accuracy")
plt.plot(history.history['accuracy'],label="accuracy")

plt.legend()
plt.show()