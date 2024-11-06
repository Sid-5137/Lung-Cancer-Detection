import os
import logging
from tqdm import tqdm
import tensorflow as tf
from keras.models import Model
from keras.applications import ResNet50, MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Concatenate

# Suppress warnings and TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Set up logging
logging.basicConfig(filename='logs/training_log.txt', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting training for Lung Cancer Detection Model")

# Directory paths
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Image parameters
input_shape = (224, 224, 3)
batch_size = 32
num_classes = 3  # Benign, Malignant, Normal

# Load ResNet50 and MobileNet as base models
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Rename layers to avoid conflicts
for layer in resnet_base.layers:
    layer._name = "resnet_" + layer.name
for layer in mobilenet_base.layers:
    layer._name = "mobilenet_" + layer.name

# Freeze base models
resnet_base.trainable = False
mobilenet_base.trainable = False

# Concatenate model outputs with Global Average Pooling and dense layers for classification
resnet_output = GlobalAveragePooling2D()(resnet_base.output)
mobilenet_output = GlobalAveragePooling2D()(mobilenet_base.output)
merged = Concatenate()([resnet_output, mobilenet_output])

x = Dense(128, activation='relu')(merged)
x = Dense(64, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

# Create and compile the final model
model = Model(inputs=[resnet_base.input, mobilenet_base.input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data generators
datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = datagen.flow_from_directory(train_dir, target_size=(224, 224),
                                              batch_size=batch_size, class_mode='categorical')
val_generator = datagen.flow_from_directory(val_dir, target_size=(224, 224),
                                            batch_size=batch_size, class_mode='categorical')
test_generator = datagen.flow_from_directory(test_dir, target_size=(224, 224),
                                             batch_size=batch_size, class_mode='categorical')

# Custom training loop with progress bar and logging
epochs = 20
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_progress_bar = tqdm(total=steps_per_epoch, desc="Training", unit="batch")
    val_progress_bar = tqdm(total=validation_steps, desc="Validation", unit="batch")

    # Reset metrics at the beginning of each epoch
    model.reset_metrics()
    total_loss, total_accuracy = 0, 0
    val_total_loss, val_total_accuracy = 0, 0

    # Training Loop
    for batch, (x_batch, y_batch) in enumerate(train_generator):
        loss, acc = model.train_on_batch([x_batch, x_batch], y_batch)
        train_progress_bar.set_postfix({"loss": f"{loss:.4f}", "accuracy": f"{acc:.4f}"})
        train_progress_bar.update(1)
        total_loss += loss
        total_accuracy += acc
        if batch >= steps_per_epoch - 1:
            break

    train_progress_bar.close()
    avg_loss = total_loss / steps_per_epoch
    avg_accuracy = total_accuracy / steps_per_epoch

    # Validation Loop
    for batch, (x_val, y_val) in enumerate(val_generator):
        val_loss, val_acc = model.test_on_batch([x_val, x_val], y_val)
        val_progress_bar.set_postfix({"val_loss": f"{val_loss:.4f}", "val_accuracy": f"{val_acc:.4f}"})
        val_progress_bar.update(1)
        val_total_loss += val_loss
        val_total_accuracy += val_acc
        if batch >= validation_steps - 1:
            break

    val_progress_bar.close()
    avg_val_loss = val_total_loss / validation_steps
    avg_val_accuracy = val_total_accuracy / validation_steps

    # Log epoch metrics to the file
    logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, "
                 f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")

# Evaluate the model on the test set and display results
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Log final test results
logging.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# Save the trained model
model.save('lung_cancer_detection_model.keras')
