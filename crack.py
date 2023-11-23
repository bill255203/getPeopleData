import cv2
import numpy as np
from PIL import Image
from tensorflow.keras import datasets, models, layers

def preprocess_image(image_path, num_digits=6):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Assuming the digits are evenly spaced across the image
    # Split the image into `num_digits` parts
    width = img.shape[1]
    split_width = width // num_digits
    digit_images = []

    for i in range(num_digits):
        # Extract each digit sub-image
        digit = img[:, i*split_width:(i+1)*split_width]

        # Resize to 28x28 and invert colors
        resized_digit = cv2.resize(digit, (28, 28))
        resized_digit = np.expand_dims(resized_digit, axis=-1)  # Add channel dimension
        resized_digit = resized_digit.astype('float32') / 255  # Normalize

        digit_images.append(resized_digit)

    return np.array(digit_images)


def load_data():
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
    return train_images, train_labels, test_images, test_labels

def build_model():
    # Build the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # Add more layers...
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    return model

def train_model(model, train_images, train_labels, test_images, test_labels):
    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# This makes sure this code block doesn't run when imported
if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_data()
    model = build_model()
    train_model(model, train_images, train_labels, test_images, test_labels)
    model.save('my_model.keras')
