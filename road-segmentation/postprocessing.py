# Patchwise Road Segmentation for Aerial Images with CNN
# Emmanouil Angelis, Spyridon Angelopoulos, Georgios Touloupas
# Group 5: Google Maps Team
# Department of Computer Science, ETH Zurich, Switzerland
# Computational Intelligence Lab

# This script is used for postprocessing (model ensemble, CRFs)

import os
import shutil
import argparse
import glob
import pickle
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.io import imread, imsave
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

PIXEL_DEPTH = 255
NUM_LABELS = 2
IMG_PATCH_SIZE = 16

def create_empty_dir(dir):
    os.makedirs(dir, exist_ok=True)
    shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

def expand_patches(softmax_outputs, axes=(0,1)):
    expanded = softmax_outputs.repeat(IMG_PATCH_SIZE, axis=axes[0]).repeat(IMG_PATCH_SIZE, axis=axes[1])
    return expanded

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg/np.max(rimg)*PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def patch_to_label(patch):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def calculate_patch_predictions(img):
    predictions = []
    for i in range(0, img.shape[0], IMG_PATCH_SIZE):
        row_predictions = []
        for j in range(0, img.shape[1], IMG_PATCH_SIZE):
            patch = img[i:i + IMG_PATCH_SIZE, j:j + IMG_PATCH_SIZE]
            row_predictions.append(patch_to_label(patch))
        predictions.append(row_predictions)
    return np.asarray(predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_dir", type=str, default="test_set_images", 
                        help="directory containing the test dataset")
    parser.add_argument("--softmax_outputs_dir", type=str, default="softmax_outputs",
                        help="directory containing the softmax outputs of different models for the test set")
    parser.add_argument("--ensemble_dir", type=str, default="predictions_ensemble", 
                        help="directory to store the ensemble predictions")
    parser.add_argument("--gaussian_blur_sigma", type=int, default=7, 
                        help="Gaussian blur radius to be applied on the ensemble probabilities")
    parser.add_argument("--crf_dir", type=str, default="predictions_crf", 
                        help="directory to store the CRF predictions")
    parser.add_argument("--crf_inference_steps", type=int, default=1, 
                        help="CRF inference steps")
    parser.add_argument("--gaussian_sxy", type=int, default=3, 
                        help="Pairwise Gaussian potential sxy features")
    parser.add_argument("--bilateral_sxy", type=int, default=20, 
                        help="Pairwise bilateral potential sxy features")
    parser.add_argument("--bilateral_srgb", type=int, default=20, 
                        help="Pairwise bilateral potential srgb features")
    args = parser.parse_args()
    
    create_empty_dir(args.ensemble_dir)
    create_empty_dir(args.crf_dir)
    
    print("Loading test set")
    imgs = {}
    for test_img in glob.glob(os.path.join(args.test_dir, "*.png")):
        imgs[test_img.split("test_")[-1]] = imread(test_img)
    img = next(iter(imgs.values()))
    H = img.shape[0]
    W = img.shape[1]
    
    print("Loading softmax outputs for all models")
    models = {}
    for softmax_outputs_file in glob.glob(os.path.join(args.softmax_outputs_dir, "*.pkl")):
        f = open(softmax_outputs_file, "rb")
        softmax_outputs = pickle.load(f)
        f.close()
        new_softmax_outputs = {}
        for test_img, softmax_output in softmax_outputs.items():
            new_softmax_outputs[test_img.split("test_")[-1]] = np.asarray(softmax_output).reshape((H//IMG_PATCH_SIZE, W//IMG_PATCH_SIZE, NUM_LABELS)).transpose([2, 1, 0])
        models[softmax_outputs_file] = new_softmax_outputs
    
    print("Calculating ensemble predictions")
    ensemble_softmax_outputs = {}
    blurred_softmax_outputs = {}
    for softmax_outputs_file, softmax_outputs in models.items():
        for test_img, softmax_output in softmax_outputs.items():
            if test_img not in ensemble_softmax_outputs:
                ensemble_softmax_outputs[test_img] = []
            ensemble_softmax_outputs[test_img] += [softmax_output]
    for test_img, softmax_output in ensemble_softmax_outputs.items():
        ensemble_softmax_outputs[test_img] = expand_patches(np.asarray(softmax_output).mean(axis=0), axes=(1,2))
        imsave(args.ensemble_dir + "/softmax_ensemble_" + test_img, img_float_to_uint8(ensemble_softmax_outputs[test_img][1]))
        ensemble_prediction = ensemble_softmax_outputs[test_img].argmax(axis=0)
        imsave(args.ensemble_dir + "/prediction_test_" + test_img, img_float_to_uint8(ensemble_prediction))
        blurred_probabilities = gaussian_filter(ensemble_softmax_outputs[test_img][1], sigma=args.gaussian_blur_sigma)
        imsave(args.ensemble_dir + "/blurred_softmax_ensemble_" + test_img, img_float_to_uint8(blurred_probabilities))
        blurred_softmax_outputs[test_img] = np.asarray([1 - blurred_probabilities, blurred_probabilities])
    
    print("CRF postprocessing")
    for test_img, softmax_output in blurred_softmax_outputs.items():
        d = dcrf.DenseCRF2D(W, H, NUM_LABELS)
        # Get unary potentials (neg log probability)
        U = unary_from_softmax(softmax_output)
        U = np.ascontiguousarray(U)
        d.setUnaryEnergy(U)
        # This potential enforces more spatially consistent segmentations
        d.addPairwiseGaussian(sxy=args.gaussian_sxy, 
                              compat=3, 
                              kernel=dcrf.DIAG_KERNEL, 
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This potential uses local color features to refine the segmentation
        d.addPairwiseBilateral(sxy=args.bilateral_sxy, 
                               srgb=args.bilateral_srgb, 
                               rgbim=imgs[test_img], 
                               compat=5, 
                               kernel=dcrf.DIAG_KERNEL, 
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        # Run inference steps
        Q = d.inference(args.crf_inference_steps)
        crf_probabilities = np.asarray(Q).reshape((NUM_LABELS, H, W))
        imsave(args.crf_dir + "/pixelwise_probabilities_" + test_img, img_float_to_uint8(crf_probabilities[1]))
        # Pixelwise prediction
        crf_prediction = crf_probabilities.argmax(axis=0)
        imsave(args.crf_dir + "/pixelwise_prediction_" + test_img, img_float_to_uint8(crf_prediction))
        # Per patch prediction
        crf_prediction_patch = expand_patches(calculate_patch_predictions(crf_probabilities[1]))
        imsave(args.crf_dir + "/patch_probabilities_prediction_" + test_img, img_float_to_uint8(crf_prediction_patch))
        crf_probabilities_patch = expand_patches(calculate_patch_predictions(crf_prediction))
        imsave(args.crf_dir + "/prediction_test_" + test_img, img_float_to_uint8(crf_probabilities_patch))
