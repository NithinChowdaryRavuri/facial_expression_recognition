import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Define data directories (adjust paths if necessary)
train_dir = '/Users/nithinchowdaryravuri/Desktop/deep_learning/archive/train'
test_dir = '/Users/nithinchowdaryravuri/Desktop/deep_learning/archive/test'

# Define the function to preprocess an image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((64, 64))  # Resized to 64x64 as specified in input_shape
    image = np.array(image)
    image = image.astype('float32')
    image /= 255.0
    image = (image - 0.5) * 2.0  # Apply contrast enhancement
    return image

# Load training and test data
train_data = []
test_data = []

for class_name in os.listdir(train_dir):
    for image_file in os.listdir(os.path.join(train_dir, class_name)):
        image_path = os.path.join(train_dir, class_name, image_file)
        image = preprocess_image(image_path)
        label = class_name  # Extract label from class name
        train_data.append((image, label))  # Append as a tuple (image, label)

for class_name in os.listdir(test_dir):
    if not class_name.startswith('.'):
        for image_file in os.listdir(os.path.join(test_dir, class_name)):
            image_path = os.path.join(test_dir, class_name, image_file)
            image = preprocess_image(image_path)
            label = class_name  # Extract label from class name
            test_data.append((image, label))  # Append as a tuple (image, label)

# Separate features and labels from the list of tuples
train_images, train_labels = zip(*train_data)
test_images, test_labels = zip(*test_data)

# Define a dictionary to map string labels to integers
label_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6  # Add the missing label mapping
}

# Convert string labels to integers using the label_map
train_labels = np.array(list(map(lambda x: label_map[x], train_labels)))
test_labels = np.array(list(map(lambda x: label_map[x], test_labels)))

# Convert labels to one-hot encoded format (if needed)
# if using categorical crossentropy loss:
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Reshape images (if necessary)
input_shape = (64, 64, 1)  # Assuming grayscale images
train_images = np.array([image.reshape(input_shape) for image in train_images])
test_images = np.array([image.reshape(input_shape) for image in test_images])


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Add a dropout layer after the convolutional layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Add a dropout layer after the convolutional layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # Add a dropout layer after the convolutional layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')  # 7 classes for FER2013
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, verbose=1)

# Initialize lists to store history
history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels), callbacks=[early_stopping,reduce_lr])
# Evaluate the model
model_loss, model_accuracy = model.evaluate(test_images, test_labels)

# Print the evaluation results
print('Loss:', model_loss)
print('Accuracy:', model_accuracy)

model.save('model.h5')