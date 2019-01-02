import tensorflow as tf
import os
import numpy as np
from TF_stixels.code.model import model_fn, params
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python import debug as tf_debug # for debugging
import glob
from PIL import Image
from tkinter import filedialog

import tkinter as tk
from tkinter.filedialog import askopenfilename

from model import model_fn, params

# params
H = 370
W = params.image_width
C = 3




#######################################
###   Creating a dataset_input_fn   ###
#######################################

'''
    dataset_input_fn - Constructs a Dataset, Iterator, and returns handles that will be called when
    Estimator requires a new batch of data. This function will be passed into 
    Estimator as the input_fn argument.

    Inputs:
        mode: string specifying whether to take the inputs from training or validation data
    Outputs:
        features: the columns of feature input returned from a dataset iterator
        labels: the columns of labels for training return from a dataset iterator
'''


def dataset_input_fn(mode, data_files):
    # Function that does the heavy lifting for constructing a Dataset
    #    depending on the current mode of execution
    dataset = load_dataset(mode,data_files)
    # Making an iterator that runs from start to finish once
    #    (the preferred type for Estimators)
    iterator = dataset.make_one_shot_iterator()
    # Consuming the next batch of data from our iterator
    features, labels = iterator.get_next()
    return features, labels


####################################
###   Constructing the Dataset   ###
####################################

'''
    load_dataset() - Loads and does all processing for a portion of the dataset specified by 'mode'.

    Inputs:
        mode: string specifying whether to take the inputs from training or validation data

    Outputs:
        dataset: the Dataset object constructed for this mode and pre-processed
'''


def load_dataset(mode, data_files):
    # Taking either the train or validation files from the dictionary we constructed above
    files = data_files[mode]
    # Created a Dataset from our list of TFRecord files
    dataset = tf.data.TFRecordDataset(files)
    # Apply any processing we need on each example in our dataset.  We
    #    will define parse next.  num_parallel_calls decides how many records
    #    to apply the parse function to at a time (change this based on your
    #    machine).
    dataset = dataset.map(parse, num_parallel_calls=2)
    # Shuffle the data if training, for validation it is not necessary
    # buffer_size determines how large the buffer of records we will shuffle
    #    is, (larger is better!) but be wary of your memory capacity.
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=1000)
    # Batch the data - you can pick a batch size, maybe 32, and later
    #    we will include this in a dictionary of other hyper parameters.
    dataset = dataset.batch(params.batch_size)
    return dataset



#######################################
###   Defining the Parse Function   ###
#######################################

def parse(serialized):
    # Define the features to be parsed out of each example.
    #    You should recognize this from when we wrote the TFRecord files!
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }
    # Parse the features out of this one record we were passed
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Format the data
    image_raw = parsed_example['image']
    image = tf.image.decode_png(image_raw, channels=3,
                                dtype=tf.uint8)  # Decode the raw bytes so it becomes a tensor with type.
    image = tf.cast(image, tf.float32)  # The type is now uint8 but we need it to be float.

    label = parsed_example['label']
    return {'image': image}, label


def visualize_pred(tfrecord_file, predictions_list, model_dir):
    # define the output folder
    model_dirname = os.path.basename(model_dir)
    output_folder = os.path.dirname(os.path.abspath(tfrecord_file)) + '/' + model_dirname
    image_name = os.path.basename(tfrecord_file)

    # reset tf graph
    tf.reset_default_graph()

    # Pipeline of dataset and iterator
    dataset = tf.data.TFRecordDataset([tfrecord_file])
    dataset = dataset.map(parse)
    iterator = dataset.make_one_shot_iterator()
    next_image_data = iterator.get_next()

    num_of_stixels = len(predictions_list)
    new_im = Image.new('RGB', (params.image_width + 5 * (num_of_stixels-1), 370))
    #new_im = Image.new('RGB', (24 + 5 * (num_of_stixels - 1), 370))
    x_offset = 0
    labels = []

    # Reconstruct the original image & labels
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        frame = 0

        for prediction in predictions_list:
            # extracting the 1st image_data from the TFRecord
            # to extract all image_data till TFRecord is exhausted use - while True
            # but need to handle the exception at the end)
            image_data = sess.run(next_image_data)
            image = image_data[0]['image']
            im = Image.fromarray(np.uint8(image))
            label = image_data[1]
            labels.append(label)
            #print('label = {}, prediction = {}'.format(label, prediction))
            new_im.paste(im, (x_offset, 0))
            x_offset += 5

        #im.show()
        #new_im.show()
        final_image = np.array(new_im)
        fig, ax = plt.subplots()
        ax.imshow(final_image)

        for index in range(num_of_stixels):
            #print('label = {}, prediction = {}'.format(labels[index], predictions_list[index]))

            plt.plot(int(params.image_width/2) + 5 * (index-1), labels[index] * 5, marker='o', markersize=5, color="blue")
            plt.plot(int(params.image_width/2) + 5 * (index-1), predictions_list[index] * 5, marker='o', markersize=4, color="red")

            #plt.plot(12 + 5 * (index-1), labels[index] * 5, marker='o', markersize=5, color="blue")
            #plt.plot(12 + 5 * (index-1), predictions_list[index] * 5, marker='o', markersize=4, color="red")
            plt.draw()

        image_save_name = image_name.replace('.tfrecord', '___Model_') + model_dirname + '.jpg'
        #image_save_name = image_name.replace('.tfrecord', '') + '_prediction_' + model_dirname + '.jpg'
        print(image_save_name)
        plt.savefig(os.path.join(output_folder, image_save_name))
        #plt.show()
        plt.close()


def main(test_dir, model_dir):

    #############################
    ###    Define test file   ###
    #############################

    # RAN - Setup logger - only displays the most important warnings
    tf.logging.set_verbosity(tf.logging.INFO)  # possible values - DEBUG / INFO / WARN / ERROR / FATAL

    # Load the model
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

    ' Create new directory to save prediction annotated images '
    model_dirname = os.path.basename(model_dir)
    # Create required folders
    if not os.path.exists(test_dir + '/' + model_dirname) and not os.path.isdir(test_dir + '/' + model_dirname):
        os.mkdir(test_dir + '/' + model_dirname)

    test_files = glob.glob(test_dir + '/*.tfrecord')

    for test_file in test_files:
        print(test_file)

        # Now make all these files accessible depending on whether we
        #    are in training ('train') or validation ('valid') mode.
        data_files = {'test' : test_file}

        # Prepare hooks for debugging
        hooks = [tf_debug.TensorBoardDebugHook(grpc_debug_server_addresses="dev:6064")]

        predictions = estimator.predict(input_fn=lambda: dataset_input_fn('test',data_files))
        #predictions = estimator.predict(input_fn=lambda: dataset_input_fn('test'), hooks = hooks)

        # Predict!
        predictions_list = []
        predictions_list = list(predictions)
        predicted_label = predictions_list[0]
        #print('prediction = {}'.format(predicted_label))

        # Visualize predictions based on single test TFrecord
        visualize_pred(test_file, predictions_list, model_dir)

    # Print tensorboard data
    print('tensorboard --logdir=' + str(model_dir) + '--port 6006 --debugger_port 6064')



if __name__ == '__main__':

    #test_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/data/Dataset_5/test'
    test_dir = '/media/vision/Datasets/Dataset_5/test'

    ' Determine the model to be used for inference '
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    model_dir = filedialog.askdirectory(initialdir='/home/dev/PycharmProjects/stixel/TF_stixels/results')
    root.destroy()
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2018-12-26_17-01-07_LR_0.001_EP_1000'

    main(test_dir, model_dir)
