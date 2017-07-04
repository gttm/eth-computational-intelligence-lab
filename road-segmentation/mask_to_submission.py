# Patchwise Road Segmentation for Aerial Images with CNN
# Emmanouil Angelis, Spyridon Angelopoulos, Georgios Touloupas
# Group 5: Google Maps Team
# Department of Computer Science, ETH Zurich, Switzerland
# Computational Intelligence Lab

# This script is used to transform prediction images into submissions for Kaggle

import os
import argparse
import numpy as np
import matplotlib.image as mpimg
import re

# Assign a label to a patch
def patch_to_label(patch):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

# Reads a single image and outputs the strings that should go into the submission file
def mask_to_submission_strings(image_filename):
    img_number = int(image_filename.split("prediction_test_")[-1].split(".png")[0])
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

#Converts images into a submission file
def masks_to_submission(submission_filename, *image_filenames):
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn in image_filenames[0:]:
            f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--predictions_dir", type=str, default="predictions_baseline_cnn", 
                        help="directory containing the predictions for the test set")
    parser.add_argument("--submission_filename", type=str, default="baseline_cnn_submission.csv", 
                        help="filename of the submission file to be created")
    args = parser.parse_args()
    
    image_filenames = []
    for i in range(1, 51):
        image_filename = args.predictions_dir + "/prediction_test_" + "%d" % i + ".png"
        print (image_filename)
        image_filenames.append(image_filename)
    masks_to_submission(args.submission_filename, *image_filenames)
