# Patchwise Road Segmentation for Aerial Images with CNN
# Emmanouil Angelis, Spyridon Angelopoulos, Georgios Touloupas
# Group 5: Google Maps Team
# Department of Computer Science, ETH Zurich, Switzerland
# Computational Intelligence Lab

# This script contains the Fully Convolutional model

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
BATCH_SIZE = 1
TRAIN_IMG_DIM = 400
TEST_IMG_DIM = 608
IMG_PATCH_SIZE = 16
MODEL_VERSION = 3
FLAGS = None

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for j in range(0, imgheight, h):
        for i in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract the images into a 4D tensor [image index, y, x, channels]
def extract_data(dir):
    imgs = [mpimg.imread(file) for file in glob.glob(os.path.join(dir, "*.png"))]
    return np.asarray(imgs)

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
    labels = np.asarray([[value_to_class(np.mean(patch)) for patch in image_patches] for image_patches in gt_patches])
    print("labels: " + str(labels.shape))
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
    for j in range(0, imgheight, h):
        for i in range(0, imgwidth, w):
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
        shape=(None, None, None, NUM_CHANNELS))
    labels_node = tf.placeholder(tf.float32, shape=(None,None, NUM_LABELS))
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)
    
    if FLAGS.use_pretrained:
        print("Load pretrained VGG model from " + FLAGS.pretrained_file)
        pretrained_dict = np.load(FLAGS.pretrained_file, encoding="latin1").item()
    
    def weight_variable(shape, name=None):
        if FLAGS.use_pretrained and name is not None:
            initial = tf.constant(pretrained_dict[name][0])
        else:
            initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape, name=None):
        if FLAGS.use_pretrained and name is not None:
            initial = tf.constant(pretrained_dict[name][1])
        else:
            initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv_layer(x, W, b):
        conv = tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME"), b)
        if FLAGS.batchnorm:
            conv = tf.layers.batch_normalization(conv, training=training)
        relu = tf.nn.relu(conv)
        return relu
    
    def pool_layer(x):
        pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        if FLAGS.batchnorm:
            pool = tf.layers.batch_normalization(pool, training=training)
        return pool
    
    def change_residual_dims(x, W):
        residual_dims = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
        if FLAGS.batchnorm:
            residual_dims = tf.layers.batch_normalization(residual_dims, training=training)
        return residual_dims

    # The following variables hold all the trainable weights
    W_1_1 = weight_variable([3, 3, NUM_CHANNELS, 64],name='conv1_1')
    W_1_2 = weight_variable([3, 3, 64, 64],name='conv1_2')
    W_2_1 = weight_variable([3, 3, 64, 128],name='conv2_1')
    W_2_2 = weight_variable([3, 3, 128, 128],name='conv2_2')
    W_3_1 = weight_variable([3, 3, 128, 256],name='conv3_1')
    W_3_2 = weight_variable([3, 3, 256, 256],name='conv3_2')
    W_3_3 = weight_variable([3, 3, 256, 256],name='conv3_3')
    W_4_1 = weight_variable([3, 3, 256, 512],name='conv4_1')
    W_4_2 = weight_variable([3, 3, 512, 512],name='conv4_2')
    W_4_3 = weight_variable([3, 3, 512, 512],name='conv4_3')
    b_1_1 = bias_variable([64],name='conv1_1')
    b_1_2 = bias_variable([64],name='conv1_2')
    b_2_1 = bias_variable([128],name='conv2_1')
    b_2_2 = bias_variable([128],name='conv2_2')
    b_3_1 = bias_variable([256],name='conv3_1')
    b_3_2 = bias_variable([256],name='conv3_2')
    b_3_3 = bias_variable([256],name='conv3_3')
    b_4_1 = bias_variable([512],name='conv4_1')
    b_4_2 = bias_variable([512],name='conv4_2')
    b_4_3 = bias_variable([512],name='conv4_3')
    W_data = weight_variable([1, 1, NUM_CHANNELS, 64])
    W_pool_1 = weight_variable([1, 1, 64, 128])
    W_pool_2 = weight_variable([1, 1, 128, 256])
    W_pool_3 = weight_variable([1, 1, 256, 512])


    if MODEL_VERSION == 0:
        W_5_1 = weight_variable([3, 3, 512, 512],name='conv5_1')
        W_5_2 = weight_variable([3, 3, 512, 512],name='conv5_2')
        W_5_3 = weight_variable([3, 3, 512, 512],name='conv5_3')

        b_5_1 = bias_variable([512],name='conv5_1')
        b_5_2 = bias_variable([512],name='conv5_2')
        b_5_3 = bias_variable([512],name='conv5_3')

        W_6 = weight_variable([1, 1, 512, NUM_LABELS])
        b_6 = bias_variable([NUM_LABELS])

        regularizers=tf.nn.l2_loss(W_5_3) + tf.nn.l2_loss(b_5_3) + tf.nn.l2_loss(W_6) + tf.nn.l2_loss(b_6)

    elif MODEL_VERSION == 1:
        W_5 = weight_variable([5, 5, 512, 512])
        b_5 = bias_variable([512])        

        W_6 = weight_variable([1, 1, 512, NUM_LABELS])
        b_6 = bias_variable([NUM_LABELS])
        regularizers=tf.nn.l2_loss(W_5) + tf.nn.l2_loss(b_5) + tf.nn.l2_loss(W_6) + tf.nn.l2_loss(b_6)
    elif MODEL_VERSION == 2:
        W_5 = weight_variable([5, 5, 512, 512])
        b_5 = bias_variable([512])        

        W_6 = weight_variable([1, 1, 512, 128])
        b_6 = bias_variable([128])

        W_7_1 = weight_variable([5, 5, 128, 128])
        W_7_2 = weight_variable([5, 5, 128, 128])

        b_7_1 = bias_variable([128])
        b_7_2 = bias_variable([128])

        W_8 = weight_variable([1, 1, 128, NUM_LABELS])
        b_8 = bias_variable([NUM_LABELS])

        regularizers=tf.nn.l2_loss(W_5) + tf.nn.l2_loss(b_5) + tf.nn.l2_loss(W_6) + tf.nn.l2_loss(b_6) + tf.nn.l2_loss(W_8) + tf.nn.l2_loss(b_8)
    elif MODEL_VERSION == 3:
        W_5 = weight_variable([5, 5, 512, 1024])
        b_5 = bias_variable([1024])        

        W_6 = weight_variable([1, 1, 1024, 128])
        b_6 = bias_variable([128])

        W_7 = weight_variable([1, 1, 128, NUM_LABELS])
        b_7 = bias_variable([NUM_LABELS])
        regularizers=tf.nn.l2_loss(W_5) + tf.nn.l2_loss(b_5) + tf.nn.l2_loss(W_6) + tf.nn.l2_loss(b_6) + tf.nn.l2_loss(W_7) + tf.nn.l2_loss(b_7)

    
    # The model definition
    def model(data,pred_phase=False):
        if pred_phase:
            image_dim = TEST_IMG_DIM
        else:
            image_dim = TRAIN_IMG_DIM

        if FLAGS.batchnorm:
            data = tf.layers.batch_normalization(data, training=training)
        if not FLAGS.residual:
            # VGG architecture
            conv_1_1 = conv_layer(data, W_1_1, b_1_1)
            conv_1_2 = conv_layer(conv_1_1, W_1_2, b_1_2)
            
            pool_1 = pool_layer(conv_1_2)
            
            conv_2_1 = conv_layer(pool_1, W_2_1, b_2_1)
            conv_2_2 = conv_layer(conv_2_1, W_2_2, b_2_2)
            
            pool_2 = pool_layer(conv_2_2)
            
            conv_3_1 = conv_layer(pool_2, W_3_1, b_3_1)
            conv_3_2 = conv_layer(conv_3_1, W_3_2, b_3_2)
            conv_3_3 = conv_layer(conv_3_2, W_3_3, b_3_3)
            
            pool_3 = pool_layer(conv_3_3)
            
            conv_4_1 = conv_layer(pool_3, W_4_1, b_4_1)
            conv_4_2 = conv_layer(conv_4_1, W_4_2, b_4_2)
            conv_4_3 = conv_layer(conv_4_2, W_4_3, b_4_3)
            
            pool_4 = pool_layer(conv_4_3)
        else:
            # VGG architecture with dense residual connections
            data_res = change_residual_dims(data, W_data)
            
            conv_1_1 = conv_layer(data, W_1_1, b_1_1)
            conv_1_2 = conv_layer(data_res + conv_1_1, W_1_2, b_1_2)
            
            pool_1 = pool_layer(data_res + conv_1_1 + conv_1_2)
            pool_1_res = change_residual_dims(pool_1, W_pool_1)
            
            conv_2_1 = conv_layer(pool_1, W_2_1, b_2_1)
            conv_2_2 = conv_layer(pool_1_res + conv_2_1, W_2_2, b_2_2)
            
            pool_2 = pool_layer(pool_1_res + conv_2_1 + conv_2_2)
            pool_2_res = change_residual_dims(pool_2, W_pool_2)
            
            conv_3_1 = conv_layer(pool_2, W_3_1, b_3_1)
            conv_3_2 = conv_layer(pool_2_res + conv_3_1, W_3_2, b_3_2)
            conv_3_3 = conv_layer(pool_2_res + conv_3_1 + conv_3_2, W_3_3, b_3_3)
            
            pool_3 = pool_layer(pool_2_res + conv_3_1 + conv_3_2 + conv_3_3)
            pool_3_res = change_residual_dims(pool_3, W_pool_3)
            
            conv_4_1 = conv_layer(pool_3, W_4_1, b_4_1)
            conv_4_2 = conv_layer(pool_3_res + conv_4_1, W_4_2, b_4_2)
            conv_4_3 = conv_layer(pool_3_res + conv_4_1 + conv_4_2, W_4_3, b_4_3)
            
            pool_4 = pool_layer(pool_3_res + conv_4_1 + conv_4_2 + conv_4_3)

        if MODEL_VERSION == 0:
            conv_5_1 = conv_layer(pool_4, W_5_1, b_5_1)
            conv_5_2 = conv_layer(conv_5_1, W_5_2, b_5_2)
            conv_5_3 = conv_layer(conv_5_2, W_5_3, b_5_3)
            out = tf.nn.bias_add(tf.nn.conv2d(conv_5_3, W_6, strides=[1, 1, 1, 1], padding='SAME'), b_6)
        elif MODEL_VERSION == 1:
            conv_5 = conv_layer(pool_4, W_5, b_5)
            out = tf.nn.bias_add(tf.nn.conv2d(conv_5, W_6, strides=[1, 1, 1, 1], padding='SAME'), b_6)
        elif MODEL_VERSION == 2:
            conv_5 = conv_layer(pool_4, W_5, b_5)
            conv_6 = conv_layer(conv_5, W_6, b_6)
            conv_7_1 = conv_layer(conv_6, W_7_1, b_7_1)
            conv_7_2 = conv_layer(conv_7_1, W_7_2, b_7_2)
            out = tf.nn.bias_add(tf.nn.conv2d(conv_7_2, W_8, strides=[1, 1, 1, 1], padding='SAME'), b_8)
        elif MODEL_VERSION == 3:
            conv_5 = conv_layer(pool_4, W_5, b_5)
            conv_6 = conv_layer(conv_5, W_6, b_6)
            out = tf.nn.bias_add(tf.nn.conv2d(conv_6, W_7, strides=[1, 1, 1, 1], padding='SAME'), b_7)

        out=tf.reshape(out,[-1,(image_dim//16)**2,2])
        return out
    
    # Cross-entropy loss
    logits = model(data_node)
    logits_test = model(data_node,True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels_node))
    
    # L2 regularization for the final conv layers
    loss += 5e-4 * regularizers

    # Use Adam for the optimization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    
    # Predictions for the minibatch, validation set and test set
    model_predictions = tf.nn.softmax(logits)
    model_predictions_test = tf.nn.softmax(logits_test)
    
    # Get prediction for given input image
    def get_prediction(img, sess):
        output_prediction = sess.run(model_predictions_test, feed_dict={data_node: [img],keep_prob: 1.0, training: False}).tolist()
        softmax_output = np.asarray(output_prediction).reshape([-1,2])
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
            log_file = open("cnn_fully_convolutional_log.txt", "w")
            training_indices = range(train_size)
            max_f1_score = 0
            min_loss_validation = float("inf")
            steps_with_no_improvement = 0
            global_step = 1
            loss_training = 0
            termination_flag = False
            validation_image_labels = np.reshape(np.argmax(validation_labels, 2), [-1, (TRAIN_IMG_DIM//IMG_PATCH_SIZE)**2])
            for epoch in range(1, FLAGS.max_epochs + 1):
                # Permute training indices
                perm_indices = np.random.permutation(training_indices)
                for step in range(train_size//BATCH_SIZE):
                    offset = (step*BATCH_SIZE)%(train_size - BATCH_SIZE)
                    batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]
                    batch_data = train_data[batch_indices, :, :, :]
                    batch_labels = train_labels[batch_indices]
                    # Train on one batch of training data
                    feed_dict = {data_node: batch_data, labels_node: batch_labels, keep_prob: FLAGS.dropout, training: True}
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
                            
                            feed_dict = {data_node: batch_data, labels_node: batch_labels, keep_prob: 1.0, training: False}
                            l, predictions = sess.run([loss, model_predictions], feed_dict=feed_dict)
                            loss_validation += l
                            # Calculate class predictions from probabilities
                            max_predictions += np.argmax(predictions, 2).tolist()
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
    parser.add_argument("--predictions_dir", type=str, default="predictions_cnn_fully_convolutional", 
                        help="directory to store the predictions")
    parser.add_argument("--model_dir", type=str, default="model_cnn_fully_convolutional", 
                        help="directory containing the model")
    parser.add_argument("--predictions_only", action="store_true", 
                        help="compute predictions for test set without re-training")
    parser.add_argument("--max_epochs", type=int, default=100, 
                        help="maximum number of epochs")
    parser.add_argument("--recording_steps", type=int, default=720, 
                        help="number of batches trained between loss report outputs")
    parser.add_argument("--early_stopping_steps", type=int, default=5, 
                        help="number of recording steps without improvement to stop training")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="learning rate")
    parser.add_argument("--use_pretrained", action="store_true", 
                        help="use a pretrained VGG network for initialization")
    parser.add_argument("--pretrained_file", type=str, default="vgg16.npy", 
                        help="file containing the pretrained VGG network")
    parser.add_argument("--batchnorm", action="store_true", 
                        help="perform batch normalization in every layer")
    parser.add_argument("--residual", action="store_true", 
                        help="add dense residual connections among pooling layers")
    parser.add_argument("--dropout", type=float, default=1.0, 
                        help="dropout keep probability")
    FLAGS = parser.parse_args()
    
    tf.app.run()


