# Import necessary library
import os
import shutil
import numpy as np
import pandas as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import yaml
from PIL import Image
from collections import deque
from ultralytics import YOLO

# Configure the visual appearance of Seaborn plots
sns.set(rc={'axes.facecolor': '#ffe4de'}, style='darkgrid')

# Load the pre-trained YOLOv8 nano segmentation model
model = YOLO('yolov8n-seg.pt') 

# Define the dataset_path
dataset_path = 'Pothole_Segmentation_YOLOv8'

# Set the path to the YAML file
yaml_file_path = os.path.join(dataset_path, 'data.yaml')

# Load and print the contents of the YAML file
with open(yaml_file_path, 'r') as file:
    yaml_content = yaml.load(file, Loader=yaml.FullLoader)
    print(yaml.dump(yaml_content, default_flow_style=False))

# Define the paths for training and validation datasets
train_image_path = os.path.join(dataset_path, 'train', 'images')
validation_image_path = os.path.join(dataset_path, 'valid', 'images')

# Initialize counters for the number of images
num_train_images = 0
num_validation_images = 0

# Initialize sets to hold the unique sizes of images
train_image_sizes = set()
validation_image_sizes = set()

# Check train images sizes and count
for filename in os.listdir(train_image_path):
    if filename.endswith(".jpg"):
        num_train_images += 1
        image_path = os.path.join(train_image_path, filename)
        with Image.open(image_path) as img:
            train_image_sizes.add(img.size)

# Check validation images sizes and count
for filename in os.listdir(validation_image_path):
    if filename.endswith('.jpg'):
        num_validation_images += 1
        image_path = os.path.join(validation_image_path, filename)
        with Image.open(image_path) as img:
            train_image_sizes.add(img.size)

# Print the results
print(f"Number of training images: {num_train_images}")
print(f"Number of validation images: {num_validation_images}")

# Check if all images in training set have the same size
if len(train_image_sizes) == 1:
    print(f"All training images have the same size: {train_image_sizes.pop()}")
else:
    print("Training images have varying sizes.")

# Check if all images in validation set have the same size
if len(validation_image_sizes) == 1:
    print(f"All validation images have the same size: {validation_image_sizes.pop()}")
else:
    print("Validation images have varying sizes.")

# Set the seed for the random number generator 
random.seed(0)

# Create a list of image files
image_files = [f for f in os.listdir(train_image_path) if f.endswith(".jpg")]

# Randomly select 15 images
random_images = random.sample(image_files, 15)

# Create a new figure
plt.figure(figsize=(19, 12))

# Loop through each image and display it in a 3x5 grid
for i, image_files in enumerate(random_images):
    image_path = os.path.join(train_image_path, image_files)
    image = Image.open(image_path)
    plt.subplot(3, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')

# Add a suptitle
plt.suptitle('Random Selection of Dataset Images', fontsize=24)

# Show the plot 
plt.tight_layout()
plt.show()

# Deleting unnecessary variable to free up memory
del image_files

