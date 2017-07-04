# ETH Computational Intelligence Lab Project (Spring 2017)
## Road extraction from aerial images
For this project, we are provided with a set of satellite/aerial images acquired from GoogleMaps, along with the ground-truth images where each pixel is labeled as road or background. Our goal is to train a classifier to segment roads in these images, assigning a label to every 16x16 patch.

We propose a Convolutional Neural Network that is based on the VGG architecture and Deep Residual Learning for the task of road segmentation of aerial GoogleMaps images. Regarding the input, a subregion of the original image is taken, having at its center the specific patch which we want to classify. We experiment with and apply many successful techniques for faster training and regularization, like transfer learning, batch normalization and dropout. 
