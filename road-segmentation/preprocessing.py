# Patchwise Road Segmentation for Aerial Images with CNN
# Emmanouil Angelis, Spyridon Angelopoulos, Georgios Touloupas
# Group 5: Google Maps Team
# Department of Computer Science, ETH Zurich, Switzerland
# Computational Intelligence Lab

# This script is used for preprocessing (training/validation split, dataset augmentation)

import os
import shutil
from distutils.dir_util import copy_tree
import glob
import argparse
import random
from PIL import Image

def augment_dataset(input_dir, output_dir, rotate_45):
    """
        Args:
            input_dir:          String, directory containing the dataset
            output_dir:         String, directory to store the augmented dataset
            rotate_45:          Boolean, include images rotated by 45 degrees
        
        Augments the dataset by performing rotations and mirroring
    """
    # Rotate 45 degrees
    for file in glob.glob(os.path.join(input_dir, "*.png")):
        print(file)
        img = Image.open(file)
        output_file = file.replace(input_dir, output_dir)
        img.save(output_file)
        if rotate_45:
            rot45 = img.rotate(45, resample=Image.BICUBIC).crop(box=(60, 60, 340, 340)).resize((400, 400), resample=Image.LANCZOS)
            rot45.save(output_file.replace(".", "_rot45."))
    
    # Rotate 90, 180 and 270 degrees
    for file in glob.glob(os.path.join(output_dir, "*.png")):
        print(file)
        img = Image.open(file)
        rot90 = img.rotate(90)
        rot180 = img.rotate(180)
        rot270 = img.rotate(270)
        rot90.save(file.replace(".", "_rot90."))
        rot180.save(file.replace(".", "_rot180."))
        rot270.save(file.replace(".", "_rot270."))
    
    # Mirror
    for file in glob.glob(os.path.join(output_dir, "*.png")):
        print(file)
        img = Image.open(file)
        flipLR = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipLR.save(file.replace(".", "_flipLR."))
    
def sample_validation(augmented_dir, validation_dir, samples):
    """
        Args:
            augmented_dir:      String, directory containing the augmented dataset
            validation_dir:     String, directory to store the validation dataset
            samples:            List[int], indexes of samples used for the validation dataset
        
        Splits the augmented dataset into training and validation sets
    """
    files = glob.glob(os.path.join(augmented_dir, "*.png"))
    for i in samples:
        new_file = validation_dir + "/" + os.path.basename(files[i])
        print(new_file)
        shutil.move(files[i], new_file)
    
def create_empty_dir(dir):
    """
        Args:
            dir:                String, directory to create
        
        Creates an empty directory
    """
    os.makedirs(dir, exist_ok=True)
    shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, default="training_original", 
                        help="directory containing the original dataset")
    parser.add_argument("--validation_percentage", type=float, default=0.1, 
                        help="percentage of data used for validation")
    parser.add_argument("--augment", action="store_true", 
                        help="augment the dataset with rotations and mirroring")
    parser.add_argument("--rotate_45", action="store_true", 
                        help="include images rotated by 45 degrees")
    args = parser.parse_args()
    
    if args.augment:
        if args.rotate_45:
            output_training_dir = "training_augmented_45"
            output_validation_dir = "validation_augmented_45"
        else:
            output_training_dir = "training_augmented"
            output_validation_dir = "validation_augmented"
            
        # Perform preprocessing
        create_empty_dir(output_training_dir + "/images")
        augment_dataset(args.data_dir + "/images", output_training_dir + "/images", args.rotate_45)
        
        create_empty_dir(output_training_dir + "/groundtruth")
        augment_dataset(args.data_dir + "/groundtruth", output_training_dir + "/groundtruth", args.rotate_45)
        
        # Training-Validation split
        random.seed(42)
        N = len(glob.glob(os.path.join(output_training_dir + "/images", "*.png")))
        samples = random.sample(range(N), int(args.validation_percentage*N))
        
        create_empty_dir(output_validation_dir + "/images")
        sample_validation(output_training_dir + "/images", output_validation_dir + "/images", samples)
        
        create_empty_dir(output_validation_dir + "/groundtruth")
        sample_validation(output_training_dir + "/groundtruth", output_validation_dir + "/groundtruth", samples)
    else:
        copy_tree(args.data_dir, "training")
        # Training-Validation split
        random.seed(42)
        N = len(glob.glob(os.path.join("training/images", "*.png")))
        samples = random.sample(range(N), int(args.validation_percentage*N))
        
        create_empty_dir("validation/images")
        sample_validation("training/images", "validation/images", samples)
        
        create_empty_dir("validation/groundtruth")
        sample_validation("training/groundtruth", "validation/groundtruth", samples)
