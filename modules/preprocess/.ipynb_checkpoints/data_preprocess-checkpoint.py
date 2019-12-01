import os
import argparse
import random
import cv2
from imutils import paths

def preprocess_images(files, image_dim, output_dir, label):
    '''
    Load files, crop to consistent size, and save to respective folder
    '''
    # Make class directory
    class_directory = '{}/{}'.format(output_dir, label)
    if not os.path.exists(class_directory):
        os.makedirs(class_directory)

    # Iterate through files
    for f in files:
        temp = f.split('/')
        output_file = '{}/{}/{}'.format(output_dir, label, temp[-1])
        try:
            image = cv2.imread(f)
            image = cv2.resize(image, (image_dim, image_dim))
            cv2.imwrite(output_file, image)
            print('Cropping image: {}'.format(output_file))
        except:
            print('Removing corrupted file: {}'.format(output_file))

# Define arguments
parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--raw_data_dir', type=str, help='Directory where raw data is stored')
parser.add_argument('--image_dim', type=int, help='Image dimension to be cropped to')
parser.add_argument('--train_data_dir', type=str, help='Directory to output the processed training data')
parser.add_argument('--valid_data_dir', type=str, help='Directory to output the processed valid data')
parser.add_argument('--test_data_dir', type=str, help='Directory to output the processed test data')
args = parser.parse_args()

# Get arguments from parser
raw_data_dir = args.raw_data_dir
image_dim = args.image_dim
train_data_dir = args.train_data_dir
valid_data_dir = args.valid_data_dir
test_data_dir = args.test_data_dir

# Make train, valid, test directories
if not os.path.exists(train_data_dir):
    os.makedirs(train_data_dir)

if not os.path.exists(valid_data_dir):
    os.makedirs(valid_data_dir)

if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

# Get all the classes that have been sorted into directories from previous step
classes = os.listdir(raw_data_dir)

for label in classes:

    # Get and shuffle files
    image_files = list(paths.list_images('{}/{}'.format(raw_data_dir, label)))
    random.shuffle(image_files)

    # Split into train, valid, test sets
    num_images = len(image_files)
    train_files = image_files[0:int(num_images*0.7)]
    valid_files = image_files[int(num_images*0.7):int(num_images*0.9)]
    test_files = image_files[int(num_images*0.9):num_images]

    # Load files, crop to consistent size, and save to respective folder
    preprocess_images(train_files, image_dim, train_data_dir, label)
    preprocess_images(valid_files, image_dim, valid_data_dir, label)
    preprocess_images(test_files, image_dim, test_data_dir, label)
