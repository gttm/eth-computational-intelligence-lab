# Patchwise Road Segmentation for Aerial Images with CNN
# Emmanouil Angelis, Spyridon Angelopoulos, Georgios Touloupas
# Group 5: Google Maps Team
# Department of Computer Science, ETH Zurich, Switzerland
# Computational Intelligence Lab

# This script contains the Baseline CNN model

import gzip
import os
import glob
import argparse
import pickle
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 2
BATCH_SIZE = 16
TRAIN_IMG_DIM = 400
IMG_PATCH_SIZE = 16

FLAGS = None

# Extract patches from a given image
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

# Extract the images into a 4D tensor [image index, y, x, channels]
# Values are rescaled from [0, 255] down to [-0.5, 0.5]
def extract_data(dir):
    imgs = [mpimg.imread(file) for file in glob.glob(os.path.join(dir, "*.png"))]
    img_patches = [img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE) for img in imgs]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]
    return np.asarray(data)

# Assign a label to a patch
def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract the labels into a 1-hot matrix [image index, label index]
def extract_labels(dir):
    gt_imgs = [mpimg.imread(file) for file in glob.glob(os.path.join(dir, "*.png"))]
    gt_patches = [img_crop(gt_img, IMG_PATCH_SIZE, IMG_PATCH_SIZE) for gt_img in gt_imgs]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])
    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg/np.max(rimg)*PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:
                l = 0
            else:
                l = 1
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH
    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def main(argv):
    # These placeholders are where training/validation/test samples and labels are fed to the graph
    data_node = tf.placeholder(
        tf.float32,
        shape=(None, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    labels_node = tf.placeholder(tf.float32, shape=(None, NUM_LABELS))
    
    # The following variables hold all the trainable weights
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(
        tf.truncated_normal([(IMG_PATCH_SIZE//2//2)**2*64, 512], stddev=0.1))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],stddev=0.1))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))
    
    # The model definition
    def model(data):
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")

        conv2 = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")
        
        # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases
        
        return out
    
    # Cross-entropy loss
    logits = model(data_node)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels_node))
    
    # L2 regularization for the fully connected parameters
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc2_weights))
    # Add the regularization term to the loss
    loss += 0.001*regularizers
    # Use Adam for the optimization
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    
    # Predictions for the minibatch, validation set and test set
    model_predictions = tf.nn.softmax(logits)
    
    # Get prediction for given input image
    def get_prediction(img, sess):
        data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        num_patches = data.shape[0]
        softmax_output = []
        for step in range(int(np.ceil(num_patches/BATCH_SIZE))):
            offset = step*BATCH_SIZE
            if offset + BATCH_SIZE < num_patches:
                batch_data = data[offset:offset + BATCH_SIZE, :, :, :]
            else:
                batch_data = data[offset:, :, :, :]
            softmax_output += sess.run(model_predictions, feed_dict={data_node: batch_data}).tolist()
        img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, softmax_output)
        
        return softmax_output, img_prediction

    # Get a concatenation of the prediction and groundtruth for given input file
    def get_prediction_image(file, sess):
        img = mpimg.imread(file)
        softmax_output, img_prediction = get_prediction(img, sess)
        return softmax_output, Image.fromarray(img_float_to_uint8(img_prediction))
    
    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(file, sess):
        img = mpimg.imread(file)
        _, img_prediction = get_prediction(img, sess)
        oimg = make_img_overlay(img, img_prediction)
        return oimg
    
    # Add ops to save and restore all the variables
    saver = tf.train.Saver()
    
    # Create a local session to run this computation
    with tf.Session() as sess:
        if not FLAGS.predictions_only:
            # Load Train Data
            print("Loading train dataset")
            train_data = extract_data(FLAGS.training_dir + "/images/")
            train_labels = extract_labels(FLAGS.training_dir + "/groundtruth/")
            
            # Load Validation Data
            print("Loading validation dataset")
            validation_data = extract_data(FLAGS.validation_dir + "/images/")
            validation_labels = extract_labels(FLAGS.validation_dir + "/groundtruth/")
            
            train_size = train_labels.shape[0]
            validation_size = validation_labels.shape[0]
            
            # Initialize all variables
            init = tf.global_variables_initializer()
            sess.run(init)
            print ("Model initialized")
            
            os.makedirs(FLAGS.model_dir, exist_ok=True)
            log_file = open("baseline_cnn_log.txt", "w")
            training_indices = range(train_size)
            max_f1_score = 0
            min_loss_validation = float("inf")
            steps_with_no_improvement = 0
            global_step = 1
            loss_training = 0
            termination_flag = False
            validation_image_labels = np.reshape(np.argmax(validation_labels, 1), [-1, (TRAIN_IMG_DIM//IMG_PATCH_SIZE)**2])
            for epoch in range(1, FLAGS.max_epochs + 1):
                # Permute training indices
                perm_indices = np.random.permutation(training_indices)
                for step in range(train_size//BATCH_SIZE):
                    offset = (step*BATCH_SIZE)%(train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # Train on one batch of training data
                    feed_dict = {data_node: batch_data, labels_node: batch_labels}
                    _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
                    loss_training += l
                    if global_step%FLAGS.recording_steps == 0:
                        loss_training = loss_training/FLAGS.recording_steps
                        
                        # Evaluate on validation set
                        loss_validation = 0
                        max_predictions = []
                        for step in range(int(np.ceil(validation_size/BATCH_SIZE))):
                            offset = step*BATCH_SIZE
                            if offset + BATCH_SIZE < validation_size:
                                batch_data = validation_data[offset:offset + BATCH_SIZE, :, :, :]
                                batch_labels = validation_labels[offset:offset + BATCH_SIZE]
                            else:
                                batch_data = validation_data[offset:, :, :, :]
                                batch_labels = validation_labels[offset:]
                            
                            feed_dict = {data_node: batch_data, labels_node: batch_labels}
                            l, predictions = sess.run([loss, model_predictions], feed_dict=feed_dict)
                            loss_validation += l
                            # Calculate class predictions from probabilities
                            max_predictions += np.argmax(predictions, 1).tolist()
                        loss_validation = loss_validation/(validation_size//BATCH_SIZE)
                        max_predictions = np.reshape(max_predictions, [-1, (TRAIN_IMG_DIM//IMG_PATCH_SIZE)**2])
                        
                        # Calculate F1 score
                        f1 = np.mean([f1_score(validation_image_labels[i], max_predictions[i]) for i in range(validation_image_labels.shape[0])])
                        
                        message = "epoch: {}, step: {}, loss_training: {:.5f}, loss_validation: {:.5f}, f1_score: {:.5f}".format(epoch, global_step, loss_training, loss_validation, f1)
                        
                        # Calculate steps with no improvement in validation loss and F1 score for early stopping
                        if loss_validation < min_loss_validation or f1 > max_f1_score:
                            # Save the model
                            save_path = saver.save(sess, FLAGS.model_dir + "/model.ckpt")
                            # Update minimum and maximum
                            if loss_validation < min_loss_validation:
                                min_loss_validation = loss_validation
                            if f1 > max_f1_score:
                                max_f1_score = f1
                            steps_with_no_improvement = 0
                            message += ", steps_with_no_improvement: {}, model saved".format(steps_with_no_improvement)
                        else:
                            steps_with_no_improvement += 1
                            message += ", steps_with_no_improvement: {}".format(steps_with_no_improvement)
                        
                        print(message)
                        log_file.write(message + "\n")
                        log_file.flush()
                        loss_training = 0
                        
                        # Early stopping condition
                        if steps_with_no_improvement == FLAGS.early_stopping_steps:
                            termination_flag = True
                            break
                    global_step += 1
                if termination_flag:
                    break
            log_file.close()
        
        # Restore the best model
        saver.restore(sess, FLAGS.model_dir + "/model.ckpt")
        print("Model restored")
        
        # Calculate predictions on test set
        os.makedirs(FLAGS.predictions_dir, exist_ok=True)
        print("Saving predictions for test set")
        softmax_outputs = {}
        for file in glob.glob(os.path.join(FLAGS.test_dir, "*.png")):
            softmax_output, pimg = get_prediction_image(file, sess)
            softmax_outputs[file] = softmax_output
            pimg.save(FLAGS.predictions_dir + "/prediction_" + os.path.basename(file))
            oimg = get_prediction_with_overlay(file, sess)
            oimg.save(FLAGS.predictions_dir + "/overlay_" + os.path.basename(file))
        # Save softmax outputs
        softmax_outputs_file = open(FLAGS.predictions_dir + "/softmax_outputs.pkl", "wb")
        pickle.dump(softmax_outputs, softmax_outputs_file)
        softmax_outputs_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--training_dir", type=str, default="training", 
                        help="directory containing the training dataset")
    parser.add_argument("--validation_dir", type=str, default="validation", 
                        help="directory containing the validation dataset")
    parser.add_argument("--test_dir", type=str, default="test_set_images", 
                        help="directory containing the test dataset")
    parser.add_argument("--predictions_dir", type=str, default="predictions_baseline_cnn", 
                        help="directory to store the predictions")
    parser.add_argument("--model_dir", type=str, default="model_baseline_cnn", 
                        help="directory containing the model")
    parser.add_argument("--predictions_only", action="store_true", 
                        help="compute predictions for test set without re-training")
    parser.add_argument("--max_epochs", type=int, default=100, 
                        help="maximum number of epochs")
    parser.add_argument("--recording_steps", type=int, default=1000, 
                        help="number of batches trained between loss report outputs")
    parser.add_argument("--early_stopping_steps", type=int, default=10, 
                        help="number of recording steps without improvement to stop training")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="learning rate")
    FLAGS = parser.parse_args()
    
    tf.app.run()
