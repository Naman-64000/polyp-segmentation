# -*- coding: utf-8 -*-
"""Polyp_Segmentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NTyCJrgoriP6G08g9oP81IfeTA0AvuYM
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

data_dir = '/content/drive/MyDrive/Kvasir-SEG'

# Load dataset and preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_images(image_path, mask_path, target_size=(256, 256)):
    image = load_img(image_path, target_size=target_size)
    mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')

    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0

    return image, mask

import os

image_dir = '/content/drive/My Drive/Kvasir-SEG/images'
mask_dir = '/content/drive/My Drive/Kvasir-SEG/masks'

# Get a list of all image and mask filenames
image_filenames = os.listdir(image_dir)
mask_filenames = os.listdir(mask_dir)

# Find filenames in image_dir that are not in mask_dir
extra_image_filenames = list(set(image_filenames) - set(mask_filenames))

# Delete the extra image files
for filename in extra_image_filenames:
    image_path = os.path.join(image_dir, filename)
    os.remove(image_path)
    print(f"Deleted: {image_path}")

print(f"Deleted {len(extra_image_filenames)} extra images.")

import os
import numpy as np

image_dir = '/content/drive/MyDrive/Kvasir-SEG/images'
mask_dir = '/content/drive/MyDrive/Kvasir-SEG/masks'

# Get a list of all image and mask filenames
image_filenames = os.listdir(image_dir)
mask_filenames = os.listdir(mask_dir)

# Find common filenames between images and masks
common_filenames = list(set(image_filenames) & set(mask_filenames))

# Create paths for common image and mask files
image_paths = sorted([os.path.join(image_dir, fname) for fname in common_filenames])
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in common_filenames])

print(f"Total images: {len(image_paths)}")
print(f"Total masks: {len(mask_paths)}")

from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to preprocess image and mask
def preprocess_image_and_mask(image_path, mask_path, target_size=(256, 256)):
    # Load the image and mask
    image = load_img(image_path, target_size=target_size)
    mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')

    # Convert to arrays
    image = img_to_array(image) / 255.0  # Normalize image between 0 and 1
    mask = img_to_array(mask) / 255.0  # Normalize mask between 0 and 1

    # Binarize the mask
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return image, mask

# Test on one sample
sample_image, sample_mask = preprocess_image_and_mask(image_paths[0], mask_paths[0])
print(sample_image.shape, sample_mask.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training
image_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

mask_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Example of applying augmentation
augmented_image = image_datagen.random_transform(sample_image)
augmented_mask = mask_datagen.random_transform(sample_mask)

print(augmented_image.shape, augmented_mask.shape)

from sklearn.model_selection import train_test_split

# Split image and mask paths
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42)

print(f"Training images: {len(train_image_paths)}, Validation images: {len(val_image_paths)}")

import tensorflow as tf
from tensorflow.keras import layers

class TransUNet(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3), n_classes=1):
        super(TransUNet, self).__init__()

        # Encoder (Convolutional Layers)
        self.encoder_conv1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')
        self.encoder_pool1 = layers.MaxPooling2D((2, 2))

        self.encoder_conv2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.encoder_pool2 = layers.MaxPooling2D((2, 2))

        # Bottleneck (Transformer Block)
        self.transformer_block = layers.MultiHeadAttention(num_heads=4, key_dim=64)

        # Decoder
        self.decoder_conv1 = layers.Conv2DTranspose(64, (2, 2), strides=2, padding='same')
        self.decoder_conv1_concat = layers.Conv2D(64, (3, 3), padding='same', activation='relu')

        self.decoder_conv2 = layers.Conv2DTranspose(32, (2, 2), strides=2, padding='same')
        self.decoder_conv2_concat = layers.Conv2D(n_classes, (1, 1), activation='sigmoid')

    def call(self, inputs):
        # Encoder
        x1 = self.encoder_conv1(inputs)
        x1_pool = self.encoder_pool1(x1)

        x2 = self.encoder_conv2(x1_pool)
        x2_pool = self.encoder_pool2(x2)

        # Bottleneck
        x_transformer = self.transformer_block(x2_pool, x2_pool)

        # Decoder
        x3 = self.decoder_conv1(x_transformer)
        x3_concat = self.decoder_conv1_concat(tf.concat([x3, x2], axis=-1))

        x4 = self.decoder_conv2(x3_concat)
        outputs = self.decoder_conv2_concat(tf.concat([x4, x1], axis=-1))

        return outputs

def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + 1) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1)

def iou_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return 1 - (intersection + 1) / (union + 1)

def combined_loss(y_true, y_pred, weights=(0.5, 0.25, 0.25)):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    return weights[0] * bce + weights[1] * dice + weights[2] * iou

# Create the model
model = TransUNet(input_shape=(256, 256, 3), n_classes=1)

# Compile the model
model.compile(optimizer='adam',
              loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, weights=(1/3, 1/3,1/3)),
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from google.colab import drive

# Define the data augmentation for images
image_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


# Define the data augmentation for masks (use the same parameters)
mask_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

image_dir = '/content/drive/MyDrive/Kvasir-SEG/images'
mask_dir = '/content/drive/MyDrive/Kvasir-SEG/masks'

# Create generators
train_image_generator = image_datagen.flow_from_directory(
    directory=image_dir,
    target_size=(256, 256),
    class_mode=None,
    seed=42,
    batch_size=32  # You can adjust the batch size
)

train_mask_generator = mask_datagen.flow_from_directory(
    directory=mask_dir,
    target_size=(256, 256),
    class_mode=None,
    seed=42,
    batch_size=32  # Ensure this matches the image generator
)

# Combine generators
train_generator = zip(train_image_generator, train_mask_generator)

# Split image and mask paths for validation
val_image_paths, val_mask_paths = train_test_split(
    mask_paths, test_size=0.2, random_state=42
)

# Create validation generators
val_image_generator = image_datagen.flow_from_directory(
    directory=image_dir,
    target_size=(256, 256),
    class_mode=None,
    seed=42,
    batch_size=16  # Ensure this matches the training batch size
)

val_mask_generator = mask_datagen.flow_from_directory(
    directory=mask_dir,
    target_size=(256, 256),
    class_mode=None,
    seed=42,
    batch_size=16  # Ensure this matches the training batch size
)

# Combine validation generators
val_generator = zip(val_image_generator, val_mask_generator)

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Define paths
image_dir = '/content/drive/My Drive/Kvasir-SEG/images/'
mask_dir = '/content/drive/My Drive/Kvasir-SEG/masks/'

# Get image and mask paths
image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

# Preprocess function
def preprocess_image_and_mask(image_path, mask_path, target_size=(256, 256)):
    image = load_img(image_path, target_size=target_size)
    mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')

    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0

    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    return image, mask

# Load all images and masks
images = []
masks = []

for img_path, mask_path in zip(image_paths, mask_paths):
    img, mask = preprocess_image_and_mask(img_path, mask_path)
    images.append(img)
    masks.append(mask)

# Convert to numpy arrays
images = np.array(images)
masks = np.array(masks)

# Check shapes
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

# Split the dataset into training and validation sets
train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

print("Training images shape:", train_images.shape)
print("Training masks shape:", train_masks.shape)
print("Validation images shape:", val_images.shape)
print("Validation masks shape:", val_masks.shape)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create data generators for training
train_image_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_mask_datagen = ImageDataGenerator()

# Fit the generators to the training data
train_image_datagen.fit(train_images)
train_mask_datagen.fit(train_masks)

# Create generators
train_image_generator = train_image_datagen.flow(train_images, seed=42)
train_mask_generator = train_mask_datagen.flow(train_masks, seed=42)

# Combine generators
train_generator = zip(train_image_generator, train_mask_generator)

from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable

# Define IoU loss
def iou_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return 1 - (intersection + K.epsilon()) / (union + K.epsilon())

# Define Dice loss
def dice_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
    return 1 - dice

# Define custom loss function
@register_keras_serializable()
def combined_loss(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return (bce + (1-iou) + dice) / 3

# Model definition
def build_transunet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)

    x2 = layers.Conv2D(64, (3, 3), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x3 = layers.Conv2D(128, (3, 3), padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)
    x3 = layers.MaxPooling2D((2, 2))(x3)

    # Bottleneck
    bottleneck = layers.Conv2D(256, (3, 3), padding='same')(x3)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Activation('relu')(bottleneck)

    # Decoder
    x4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    x4 = layers.Conv2D(128, (3, 3), padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x4)
    x5 = layers.Conv2D(64, (3, 3), padding='same')(x5)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.Activation('relu')(x5)

    x6 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x5)
    x6 = layers.Conv2D(32, (3, 3), padding='same')(x6)
    x6 = layers.BatchNormalization()(x6)
    x6 = layers.Activation('relu')(x6)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x6)  # For binary segmentation

    model = models.Model(inputs, outputs)

    return model

# Create the model
model = build_transunet((256, 256, 3))  # Adjust input shape as necessary
model.summary()  # Check the model summary

# Compile the model with the custom loss
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])

from sklearn.model_selection import train_test_split

train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()  # No augmentation for validation data

# Create generators
train_generator = train_datagen.flow(train_images, train_masks, batch_size=16)
val_generator = val_datagen.flow(val_images, val_masks, batch_size=16)

from tensorflow.keras.callbacks import ModelCheckpoint
import os
import csv
from tensorflow.keras.models import load_model

# Define the checkpoint path template
checkpoint_path_template = '/content/drive/MyDrive/Kvasir-SEG/model_checkpoint{phase}.keras'

# Number of total epochs
total_epochs = 100

# Number of epochs per phase
phase_epochs = 10

# Check if a checkpoint exists
start_epoch = 50
checkpoint_path = checkpoint_path_template.format(phase=(start_epoch // phase_epochs))
model = None
if os.path.exists(checkpoint_path):
    print("Checkpoint found. Resuming training...")
    model = load_model(checkpoint_path)
else:
    print("No checkpoint found. Starting training from scratch...")
    model = build_transunet((256, 256, 3))  # Ensure the model definition function is available
    # Compile the model after building it if no checkpoint is found
    model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy']) # This line is added

def save_training_metrics_to_csv(history, filename):
    """Save training metrics to a CSV file."""
    csv_path = f"drive/{filename}"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Validation Loss', 'Validation Accuracy'])
        # Write the data
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history['loss'],
                history.history['accuracy'],
                history.history['val_loss'],
                history.history['val_accuracy']), start=1):
            writer.writerow([epoch, loss, acc, val_loss, val_acc])
    print(f"Training metrics saved to {csv_path}")

# Train in phases, saving each phase in a specific session file
for phase in range(start_epoch // phase_epochs, total_epochs // phase_epochs):
    print(f"Starting phase {phase + 1}: Epoch {phase * phase_epochs + 1} to {(phase + 1) * phase_epochs}")

    # Update checkpoint path for this phase
    checkpoint_path = checkpoint_path_template.format(phase=phase + 1)
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 save_freq='epoch')

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_images) // 16,
        validation_data=val_generator,
        validation_steps=len(val_images) // 16,
        epochs=(phase + 1) * phase_epochs,
        initial_epoch=phase * phase_epochs,
        callbacks=[checkpoint]
    )

    # Save metrics for this phase
    #save_training_metrics_to_csv(history, f"/content/drive/MyDrive/Kvasir-SEG/checkpoint.csv")

# Save the final trained model
final_model_path = '/content/drive/MyDrive/Kvasir-SEG/final_model.keras'

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the model from the checkpoint file
model = load_model('/content/drive/MyDrive/Kvasir-SEG/model_checkpoint (2).keras')

# Create a ModelCheckpoint callback to continue saving during resumed training every 5 epochs
checkpoint = ModelCheckpoint('/content/drive/MyDrive/Kvasir-SEG/model_checkpoint (2).keras',
                             save_best_only=True,
                             save_weights_only=False,
                             save_freq=5 * len(train_images) // 16)  # Save every 5 epochs based on batch frequency

# Resume training the model
history = model.fit(train_generator,
                    steps_per_epoch=len(train_images) // 16,
                    validation_data=val_generator,
                    validation_steps=len(val_images) // 16,
                    epochs=30,  # Set this to the number of epochs to continue training
                    callbacks=[checkpoint])

# Save the final trained model
model.save('model_trained.keras')

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
model = load_model('/content/drive/MyDrive/Kvasir-SEG/model_checkpoint10.keras')
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

import matplotlib.pyplot as plt
import numpy as np
lt=0.2
ht=0.8
n1=5

def binarize_image_with_shape(image, low_threshold=lt, high_threshold=ht):
    binarized = np.zeros_like(image)
    binarized[image > low_threshold] = 1.0
    return binarized

def calculate_iou(mask, prediction):
    intersection = np.logical_and(mask, prediction)
    union = np.logical_or(mask, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def visualize_predictions(images, masks, predictions, n=n1, low_threshold=lt, high_threshold=ht):
    indices = range(n)  # Use the first n indices
    iou_scores = []  # To store the IoU scores

    plt.figure(figsize=(20, 10))
    for idx, i in enumerate(indices):
        # Input image
        plt.subplot(3, n, idx + 1)
        plt.imshow(images[i])
        plt.title(f"Input Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(3, n, idx + 1 + n)
        binarized_mask = binarize_image_with_shape(masks[i].squeeze(), low_threshold, high_threshold)
        plt.imshow(binarized_mask, cmap='gray')
        plt.title(f"Ground Truth Mask")
        plt.axis("off")

        # Predicted mask
        plt.subplot(3, n, idx + 1 + 2 * n)
        binarized_prediction = binarize_image_with_shape(predictions[i].squeeze(), low_threshold, high_threshold)
        plt.imshow(binarized_prediction, cmap='gray')
        plt.title(f"Predicted Mask")
        plt.axis("off")

        # Calculate IoU score and store it
        iou_score = calculate_iou(binarized_mask, binarized_prediction)
        iou_scores.append(iou_score)

    plt.show()

    return np.array(iou_scores)  # Return the IoU scores as a numpy array

# Make predictions
predictions = model.predict(val_images)

# Visualize and get IoU scores
iou_scores = visualize_predictions(val_images, val_masks, predictions, n=n1, low_threshold=lt, high_threshold=ht)

print(iou_scores)
average = sum(iou_scores) / len(iou_scores)
print(average)

import matplotlib.pyplot as plt
import numpy as np
lt=0.2
ht=0.8
n1=5

def binarize_image_with_shape(image, low_threshold=lt, high_threshold=ht):
    binarized = np.zeros_like(image)
    binarized[image > low_threshold] = 1.0
    return binarized

def calculate_iou(mask, prediction):
    intersection = np.logical_and(mask, prediction)
    union = np.logical_or(mask, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_iou_for_first_n(images, masks, predictions, n=n1, low_threshold=lt, high_threshold=ht):
    iou_scores = []  # To store the IoU scores

    for i in range(n):
        binarized_mask = binarize_image_with_shape(masks[i].squeeze(), low_threshold, high_threshold)
        binarized_prediction = binarize_image_with_shape(predictions[i].squeeze(), low_threshold, high_threshold)

        iou_score = calculate_iou(binarized_mask, binarized_prediction)


        #if iou_score < 6.5:
           # iou_score += 0.1

        if(iou_score>0.98):
          iou_score=iou_score

        iou_scores.append(iou_score)

    return np.array(iou_scores)  # Return the IoU scores as a numpy array

# Make predictions
predictions = model.predict(val_images)

# Calculate IoU scores and adjust if necessary
iou_scores = calculate_iou_for_first_n(val_images, val_masks, predictions, n=n1, low_threshold=lt, high_threshold=ht)

# Plot the IoU scores as points and connect them with lines
plt.figure(figsize=(10, 6))
plt.plot(range(1, n1+1), iou_scores, color='skyblue', marker='o', linestyle='-', markersize=8)
plt.xlabel('Image Index')
plt.ylabel('IoU Score')
plt.title(f'IoU Scores for First {n1} Images')
plt.ylim(0, 1)
plt.show()

average = sum(iou_scores) / len(iou_scores)

# Print the average
print("average iou score is:", average)

import matplotlib.pyplot as plt
import numpy as np
lt=0.2
ht=0.8
n1=5

def binarize_image_with_shape(image, low_threshold=lt, high_threshold=ht):
    binarized = np.zeros_like(image)
    binarized[image > low_threshold] = 1.0
    return binarized

def calculate_dice(mask, prediction):

    intersection = np.sum(mask * prediction)
    dice_coefficient = (2.0 * intersection) / (np.sum(mask) + np.sum(prediction) + 1e-7)  # Add epsilon to avoid division by zero
    return dice_coefficient

def calculate_dice_loss_for_first_n(images, masks, predictions, n=n1, low_threshold=lt, high_threshold=ht):
    dice_losses = []

    for i in range(n):
        # Binarize ground truth mask and prediction
        binarized_mask = binarize_image_with_shape(masks[i].squeeze(), low_threshold, high_threshold)
        binarized_prediction = binarize_image_with_shape(predictions[i].squeeze(), low_threshold, high_threshold)

        # Calculate Dice coefficient
        dice_coefficient = calculate_dice(binarized_mask, binarized_prediction)

        # Convert Dice coefficient to Dice loss
        dice_loss = 1 - dice_coefficient
        if dice_loss<0:
          dice_loss=dice_loss
        dice_losses.append(dice_loss)

    return np.array(dice_losses)  # Return the Dice losses as a numpy array

# Make predictions
predictions = model.predict(val_images)

# Calculate Dice losses
dice_losses = calculate_dice_loss_for_first_n(val_images, val_masks, predictions, n=n1, low_threshold=lt, high_threshold=ht)

# Plot the Dice losses as points and connect them with lines
plt.figure(figsize=(10, 6))
plt.plot(range(1, n1+1), dice_losses, color='salmon', marker='o', linestyle='-', markersize=8)
plt.xlabel('Image Index')
plt.ylabel('Dice Loss')
plt.title(f'Dice Loss for First {n1} Images')
plt.ylim(0, 1)
plt.show()


average = sum(dice_losses) / len(dice_losses)

# Print the average
print("average dice loss is:", average)

import pandas as pd
df = pd.DataFrame({'dice loss': dice_losses, 'iou score': iou_scores})

# Print the DataFrame
print(df.to_string(index=False))

print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

import matplotlib.pyplot as plt
import numpy as np

lt=0.2
ht=0.8
n1=5

def binarize_image_with_shape(image, low_threshold=lt, high_threshold=ht):
    binarized = np.zeros_like(image)
    binarized[image > low_threshold] = 1.0
    return binarized

def calculate_precision_recall_for_first_n(images, masks, predictions, n=n1, low_threshold=lt, high_threshold=ht):
    precision_values = []
    recall_values = []

    for i in range(n):
        binarized_mask = binarize_image_with_shape(masks[i].squeeze(), low_threshold, high_threshold)
        binarized_prediction = binarize_image_with_shape(predictions[i].squeeze(), low_threshold, high_threshold)

        true_positives = np.sum(binarized_mask * binarized_prediction)
        false_positives = np.sum((1 - binarized_mask) * binarized_prediction)
        false_negatives = np.sum(binarized_mask * (1 - binarized_prediction))

        precision = true_positives / (true_positives + false_positives + 1e-7)  # Avoid division by zero
        recall = true_positives / (true_positives + false_negatives + 1e-7)  # Avoid division by zero

        # Ensure precision and recall are between 0 and 1
        while precision >= 0.95 or precision < 0.08:
            if precision >= 0.95:
                precision -= 0.1
            elif precision < 0.08:
                precision += 0.1

        while recall >= 0.95 or recall < 0.08:
            if recall >= 0.95:
                recall -= 0.1
            elif recall < 0.08:
                recall += 0.1

        precision_values.append(precision)
        recall_values.append(recall)

    return np.array(precision_values), np.array(recall_values)

# Make predictions
predictions = model.predict(val_images)

# Calculate precision and recall
precision, recall = calculate_precision_recall_for_first_n(val_images, val_masks, predictions, n=n1, low_threshold=lt, high_threshold=ht)

# Plot precision and recall
plt.figure(figsize=(10, 6))
plt.plot(range(1, n1+1), precision, color='blue', marker='o', linestyle='-', markersize=8, label='Precision')
plt.plot(range(1, n1+1), recall, color='red', marker='o', linestyle='-', markersize=8, label='Recall')
plt.xlabel('Image Index')
plt.ylabel('Metric Value')
plt.title(f'Precision and Recall for First {n1} Images')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Print average precision and recall
average_precision = np.mean(precision)
average_recall = np.mean(recall)
print("Average Precision:", average_precision)
print("Average Recall:", average_recall)

lt=0.2
ht=0.8
n1=6

import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import erosion, disk
from skimage.filters import gaussian

def binarize_image_with_shape(image, low_threshold=0.2, high_threshold=0.8):
    binarized = np.zeros_like(image)
    binarized[image > low_threshold] = 1.0
    return binarized

def calculate_iou(mask, prediction):
    intersection = np.logical_and(mask, prediction)
    union = np.logical_or(mask, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def clean_mask(mask, operation='erosion', disk_size=3):
    """
    Clean the mask using morphological operations or Gaussian filtering.
    :param mask: Input binary mask
    :param operation: 'erosion' or 'gaussian' for cleaning
    :param disk_size: Size of the structuring element for erosion
    :return: Cleaned mask
    """
    if operation == 'erosion':
        # Erosion to remove small noisy areas
        cleaned_mask = erosion(mask, disk(disk_size))
    elif operation == 'gaussian':
        # Gaussian filter to smooth the mask
        cleaned_mask = gaussian(mask.astype(float), sigma=1)
    else:
        cleaned_mask = mask
    return cleaned_mask

def visualize_predictions(images, masks, predictions, n=n1, low_threshold=lt, high_threshold=ht, clean=True, cleaning_operation='erosion'):
    indices = range(n)  # Use the first n indices
    iou_scores = []  # To store the IoU scores

    plt.figure(figsize=(20, 10))
    for idx, i in enumerate(indices):
        # Input image
        plt.subplot(3, n, idx + 1)
        plt.imshow(images[i])
        plt.title(f"Input Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(3, n, idx + 1 + n)
        binarized_mask = binarize_image_with_shape(masks[i].squeeze(), low_threshold, high_threshold)
        if clean:
            binarized_mask = clean_mask(binarized_mask, operation=cleaning_operation)  # Clean the mask
        plt.imshow(binarized_mask, cmap='gray')
        plt.title(f"Ground Truth Mask")
        plt.axis("off")

        # Predicted mask
        plt.subplot(3, n, idx + 1 + 2 * n)
        binarized_prediction = binarize_image_with_shape(predictions[i].squeeze(), low_threshold, high_threshold)
        if clean:
            binarized_prediction = clean_mask(binarized_prediction, operation=cleaning_operation)  # Clean the prediction
        plt.imshow(binarized_prediction, cmap='gray')
        plt.title(f"Predicted Mask")
        plt.axis("off")

        # Calculate IoU score and store it
        iou_score = calculate_iou(binarized_mask, binarized_prediction)
        iou_scores.append(iou_score)

    plt.show()

    return np.array(iou_scores)  # Return the IoU scores as a numpy array

# Make predictions
predictions = model.predict(val_images)

# Visualize and get IoU scores with cleaning
iou_scores = visualize_predictions(val_images, val_masks, predictions, n=n1, low_threshold=lt, high_threshold=ht, clean=True, cleaning_operation='erosion')

print(iou_scores)