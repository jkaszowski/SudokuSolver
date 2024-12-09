from PIL import Image, ImageOps
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize storage for the subset
subset_images = []
subset_labels = []

# Directory to save images
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# Loop through each digit (0-9)
for digit in range(10):
    os.makedirs(f"{image_dir}/{digit}",exist_ok=True)
    # Find indices of all images of the current digit
    indices = np.where(y_train == digit)[0]

    # Select the first 10 examples for this digit
    selected_indices = indices[:20]

    # Add these examples to the subset
    subset_images.append(x_train[selected_indices])
    subset_labels.append(y_train[selected_indices])

# Combine all subsets into arrays
subset_images = np.concatenate(subset_images, axis=0)
subset_labels = np.concatenate(subset_labels, axis=0)

# Shuffle the subset to mix digits
shuffled_indices = np.random.permutation(len(subset_labels))
subset_images = subset_images[shuffled_indices]
subset_labels = subset_labels[shuffled_indices]

# Output the shapes of the subset
print("Subset Images Shape:", subset_images.shape)  # Should be (100, 28, 28)
print("Subset Labels Shape:", subset_labels.shape)  # Should be (100,)

# Save or use the subset as needed
# For example: expand dimensions to match model input requirements
subset_images = np.expand_dims(subset_images, axis=-1)  # Shape becomes (100, 28, 28, 1)
print("Updated Subset Images Shape:", subset_images.shape)  # (100, 28, 28, 1)





# Save each image with its label
for i, (image, label) in enumerate(zip(subset_images, subset_labels)):
    # Convert the image to a PIL Image object
    img = Image.fromarray(image.squeeze())  # Remove channel dimension if grayscale
    img = ImageOps.invert(img)

    # Save the image as a PNG file with the label in the filename
    img.save(os.path.join(image_dir, f"{label}/{label}_mnist_{i}.png"))

print(f"Images saved in directory: {image_dir}")
