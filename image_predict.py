

#from __future__ import absolute_import

import tensorflow as tf
import os
import numpy as np
from TF_stixels.code.model import model_fn, params
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tensorflow.python import debug as tf_debug # for debugging
import glob
from PIL import Image
from tkinter import filedialog
import tkinter as tk
import sys
import time
from TF_stixels.data.folder2TFRec2 import Frame2StxTfrecords  # NEW: use the special version to extract frame id's rather than labels!!!!!!
import cv2
import csv


# params
H = 370
C = 3

# Init CRF parameters
'''
N = 1
T = 20 #20
W_trans = 10
'''

N = 0
T = 20 #20
W_trans = 20

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

    '''
    features, label, frame_id, frame_name = iterator.get_next()
    return features, label, frame_id, frame_name
    '''

    features, frame_id = iterator.get_next()
    return features, frame_id


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
    '''
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'frame_id': tf.FixedLenFeature([], tf.int64),
        'name': tf.FixedLenFeature([], tf.string),
    }
    '''

    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'frame_id': tf.FixedLenFeature([], tf.int64),
    }

    # Parse the features out of this one record we were passed
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Format the data
    image_raw = parsed_example['image']
    image = tf.image.decode_png(image_raw, channels=3,
                                dtype=tf.uint8)  # Decode the raw bytes so it becomes a tensor with type.
    image = tf.cast(image, tf.float32)  # The type is now uint8 but we need it to be float.

    '''
    label = parsed_example['label']
    frame_id = parsed_example['frame_id']
    frame_name = parsed_example['name']    
    return {'image': image}, label, frame_id, frame_name
    '''

    frame_id = parsed_example['frame_id']
    return {'image': image}, frame_id



#############################################################
###   Visualizing predictions and creating output video   ###
#############################################################

def visualize_pred(image_in, tfrecord_file, predictions_list, image_width, out_folder, debug_image):

    image_folder = os.path.dirname(out_folder)
    image_num = 0
    plot_border_width = 2.0

    im = Image.open(image_in)
    image_size = im.size
    print('image size =  {}'.format(image_size))

    # reset tf graph
    tf.reset_default_graph()

    # Pipeline of dataset and iterator
    dataset = tf.data.TFRecordDataset([tfrecord_file])
    dataset = dataset.map(parse)
    iterator = dataset.make_one_shot_iterator()
    next_image_data = iterator.get_next()

    num_of_stixels = len(predictions_list)
    print('Number of stixels to be proceesed  {}'.format(num_of_stixels))

    # Prepare data structure for softmax predictions
    grid_x_width = int((image_width-36)/5)+1
    grid_y_width = 74
    print('grid width = {}'.format(grid_x_width))
    #grid = np.zeros((grid_y_width,grid_x_width))

    # Create data grid
    x = np.linspace(0, grid_x_width - 1, grid_x_width)*5 + 18 - 3 # reduce 3 to center the probability points
    y = np.linspace(0, grid_y_width - 1, grid_y_width)*5 - 2
    X, Y = np.meshgrid(x, y)
    print(np.shape(X), np.shape(Y))

    # Go through the TFRecord and reconstruct the images + predictions
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Init new image
        new_im = Image.new('RGB', (image_width, 370))
        grid = np.zeros((grid_y_width, grid_x_width))
        x_offset = 0
        first_time = True
        fig, ax = plt.subplots()

        # Go through all the stixels in the tfrecord file
        #for i in range(177):
        for i in range(num_of_stixels):

            image_data = sess.run(next_image_data)
            image = image_data[0]['image']
            im = Image.fromarray(np.uint8(image))
            frame_id = image_data[1]
            prediction = predictions_list[i]['classes']
            prediction_softmax = predictions_list[i]['probabilities']

            #####################################################################################
            ## Collect all image predictions into a new array tp be filtered by CRF++/CRFSuite ##
            #####################################################################################

            #label = image_data[1]
            #frame_id = image_data[2]
            #frame_name = image_data[3]

            # Collect and visualize stixels
            new_im.paste(im, (frame_id * 5, 0))
            x_offset += 5
            if debug_image:
                plt.plot(int(params.image_width / 2) + 5 * (frame_id), prediction * 5, marker='o', markersize=4, color="red")
            # visualize probabilities
            grid[:,frame_id] = prediction_softmax
            plt.draw()

        # Use CRF to find the best path
        from code.crf import viterbi
        best_path = viterbi(grid.T, N, T, W_trans)

        # Plot the CRF border line
        best_path_points = []
        for index, path in enumerate(best_path):
            best_path_points.append([int(params.image_width / 2) + index*5, path*5 + 2])
        plt.plot(np.array(best_path_points)[:,0], np.array(best_path_points)[:,1], color="blue", linewidth=plot_border_width)

        # Plot the labels
        annotation_in = image_in.replace('.jpg', '.csv')
        if os.path.exists(annotation_in):
            del_y = image_size[1] - 370
            # Init from the CSV file
            with open(annotation_in) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                label_coords = []
                for index, row in enumerate(csv_reader):
                    new_tuple = tuple(row)
                    x_coord = int(new_tuple[0])
                    y_coord = max(int(new_tuple[1]) - del_y, 0)
                    plt.plot(x_coord, y_coord, marker='o', markersize=5, color="red")
                    label_coords.append([x_coord, y_coord])

            plt.plot(np.array(label_coords)[:, 0], np.array(label_coords)[:, 1], color="red", linewidth=1.0)

        if debug_image:
            grid = np.ma.masked_array(grid, grid < .0001)
            plt.pcolormesh(X, Y, grid, norm=colors.LogNorm(), alpha = 0.75)
        plt.imshow(new_im, cmap='gray', alpha=1.0, interpolation='none')

        image_file_name = os.path.join(image_folder, 'frame' + format(image_num, '03d') + '.jpg')
        plt.savefig(image_file_name, format='jpg')
        print('saving fig to ', image_file_name)

        plt.show()
        plt.close()



######################################
### translate image ro a TFRecord ###
#####################################

def image_2_tfrec(image_in, tfrec_filename, model_stixel_width):

    start_time = time.time()
    # Create TFRec writer
    os.chdir(os.path.dirname(image_in))
    if not (glob.glob(tfrec_filename)):
        print('output file = ', tfrec_filename)
        writer = tf.python_io.TFRecordWriter(tfrec_filename)
        # parse the image and save it to TFrecord
        f_to_stx = Frame2StxTfrecords(image_in, image_in.replace('.jpg','.csv'), writer, os.path.dirname(image_in), model_stixel_width, 'test')
        f_to_stx.create_stx(False)
        writer.close()
        duration = time.time() - start_time
        print('TFRec creation took {} sec'.format(int(duration)))
    else:
        print('tfrecord already exists - skipping TFrec creation!!')

##############################################################
###                           main                         ###
##############################################################

def main(image_in, image_out_dir, image_width = 476):
    # Determine the model
    model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-28_18-57-33_EP_250'

    # If required create required folder
    if not os.path.exists(image_out_dir) and not os.path.isdir(image_out_dir):
        os.mkdir(image_out_dir)

    # Restore model.py that was saved in model directory
    os.chdir(model_dir)
    sys.path.insert(0, os.getcwd())
    if os.path.exists(model_dir + '/model_for_CRF.py'):
        from model_for_CRF import model_fn, params
    else:
        print('No model file within directory - exiting!!!!')
        os.exit()

    W = params.image_width

    # Translate the image to a TF record
    tfrec_filename = image_in.replace('.jpg','_W' + str(W) + '.tfrecord')
    print(tfrec_filename)
    image_2_tfrec(image_in, tfrec_filename, W)

    # Predict + visualize the output image
    tf.logging.set_verbosity(tf.logging.INFO)  # possible values - DEBUG / INFO / WARN / ERROR / FATAL

    # Load the model
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

    data_files = {'test': tfrec_filename}
    start_time = time.time()
    predictions = estimator.predict(input_fn=lambda: dataset_input_fn('test', data_files))

    '''
    hooks = [tf_debug.TensorBoardDebugHook(grpc_debug_server_addresses="dev:6064")]    
    predictions = estimator.predict(input_fn=lambda: dataset_input_fn('test'), hooks = hooks)
    '''
    # Print prediction duration
    duration = time.time() - start_time
    print('prediction took {}ms'.format(duration * 1000))

    # Predict!
    predictions_list = list(predictions)

    # Visualize predictions based on single test TFrecord
    visualize_pred(image_in, tfrec_filename, predictions_list, image_width, image_out_dir, debug_image=False)

    # Print tensorboard data
    print('tensorboard --logdir=' + str(model_dir) + '--port 6006')
    # print('tensorboard --logdir=' + str(model_dir) + '--port 6006 --debugger_port 6064')


    # Save output file


if __name__ == '__main__':

    from tkinter.filedialog import askopenfilename

    # Determine input image
    image_in = '/media/vision/Results/image_for_predict/frame_000142.jpg'
    image_width = 476
    if not os.path.exists(image_in):
        print('no such file')
        os.exit()

    '''
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    # image_filename = askopenfilename(initialdir='/media/vision/In Process/ForAnnotation/GC23_fixes')  # show an "Open" dialog box and return the path to the selected file
    image_in = askopenfilename(
        initialdir='/media/vision/Results/',
        filetypes=[('JPG file', '*.jpg')])  # show an "Open" dialog box and return the path to the selected file
    root.destroy()
    print('predict image - ' + image_filename)
    '''

    image_out_dir = '/media/vision/Results/image_for_predict/'

    main(image_in, image_out_dir)











