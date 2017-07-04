# Patchwise Road Segmentation for Aerial Images with CNN
# Emmanouil Angelis, Spyridon Angelopoulos, Georgios Touloupas
# Group 5: Google Maps Team
# Department of Computer Science, ETH Zurich, Switzerland
# Computational Intelligence Lab

# This script contains the Baseline Logistic model

import os
import sys
import glob
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from sklearn import linear_model
from sklearn.metrics import f1_score

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg/np.max(rimg)*255).round().astype(np.uint8)
    return rimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X

def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

if __name__ == "__main__":
    # Load the training dataset
    training_dir = "training/"
    print("Loading training images")
    image_dir = training_dir + "images/"
    files = glob.glob(os.path.join(image_dir, "*.png"))
    imgs = [load_image(file) for file in files]
    print("Loading training groundtruth")
    gt_dir = training_dir + "groundtruth/"
    files = glob.glob(os.path.join(gt_dir, "*.png"))
    gt_imgs = [load_image(file) for file in files]
    n = len(files)
    
    # Extract patches from input images
    patch_size = 16
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]
    
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    
    # Compute features for each image patch
    print("Computing features")
    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    
    # Train a logistic regression classifier
    print("Training the logistic regression classifier")
    logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
    logreg.fit(X, Y)
    
    # Load the validation dataset
    validation_dir = "validation/"
    print("Loading validation images")
    image_dir_val = validation_dir + "images/"
    files_val = glob.glob(os.path.join(image_dir_val, "*.png"))
    imgs_val = [load_image(file) for file in files_val]
    print("Loading training groundtruth")
    gt_dir_val = validation_dir + "groundtruth/"
    files = glob.glob(os.path.join(gt_dir_val, "*.png"))
    gt_imgs_val = [load_image(file) for file in files]
    n_val = len(files_val) 
    
    gt_patches_val = [img_crop(gt_imgs_val[i], patch_size, patch_size) for i in range(n_val)]
    
    X_val = [extract_img_features(files_val[i]) for i in range(n_val)]
    Y_val = [[value_to_class(np.mean(patch)) for patch in image_patches] for image_patches in gt_patches_val]
    
    # Run predictions on validation
    print("Calculating predictions for validation set")
    Z_val = [logreg.predict(Xi) for Xi in X_val]
    
    # Calculate average F1 score on validation
    f1_average = np.mean([f1_score(Y_val[i], Z_val[i]) for i in range(n_val)])
    print("F1 score for road predictions: {:.5f}".format(f1_average))
    
    # Load test dataset
    test_dir = "test_set_images/"
    files_test = glob.glob(os.path.join(test_dir, "*.png"))
    imgs_test = np.asarray([load_image(file) for file in files_test])
    X_test = [extract_img_features(file) for file in files_test]
    n_test = len(files_test)
    
    # Run predictions on test
    print("Calculating predictions for test set")
    Z_test = [logreg.predict(Xi) for Xi in X_test]
    
    # Save predictions
    w = imgs_test[0].shape[0]
    h = imgs_test[0].shape[1]
    predicted_imgs = [label_to_img(w, h, patch_size, patch_size, Zi) for Zi in Z_test]
    overlay_imgs = [make_img_overlay(imgs_test[i], predicted_imgs[i]) for i in range(n_test)]
    
    predictions_dir = "predictions_baseline_logistic/"
    os.makedirs(predictions_dir, exist_ok=True)
    print("Saving predictions for test set")
    for i in range(n_test):
        Image.fromarray(img_float_to_uint8(predicted_imgs[i])).save(predictions_dir + "prediction_" + os.path.basename(files_test[i]))
        overlay_imgs[i].save(predictions_dir + "overlay_" + os.path.basename(files_test[i]))
