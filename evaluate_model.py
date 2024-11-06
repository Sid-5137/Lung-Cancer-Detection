import os
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import logging

# Set up logging
logging.basicConfig(filename='logs/evaluation_log.txt', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting evaluation for Lung Cancer Detection Model")

# Load the saved model
model = load_model('lung_cancer_detection_model.h5')

# Directory path for test data
test_dir = 'data/test'

# Data generator for the test set
datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224),
                                             batch_size=32, class_mode='categorical')

# Custom evaluation loop with two inputs
total_loss = 0
total_accuracy = 0
num_batches = len(test_generator)

for batch, (x_batch, y_batch) in enumerate(test_generator):
    # Send the batch as two inputs to match the model's expected input format
    loss, accuracy = model.test_on_batch([x_batch, x_batch], y_batch)
    total_loss += loss
    total_accuracy += accuracy
    if batch + 1 >= num_batches:  # Stop when we reach the number of batches in the test set
        break

# Compute average test loss and accuracy
avg_test_loss = total_loss / num_batches
avg_test_accuracy = total_accuracy / num_batches

print(f"\nCustom Test Loss: {avg_test_loss:.4f}, Custom Test Accuracy: {avg_test_accuracy:.4f}")

# Log final test results
logging.info(f"Test Results - Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.4f}")
