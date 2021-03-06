
'''
*
* image_predict
*
* Purpose: the module analyzes an image using a defined model for inference, using the image_predictor class
*
* Inputs: image file name, required model
*
* Outputs: ArgMax + softmax predictions
*
* Conventions: (x=0, y=0) is the upper left corner of the image
*
* Written by: Ran Zaslavsky 10-03-2019 (framework originates from excellent https://crosleythomas.github.io/blog/)
*
'''


#from __future__ import absolute_import

import tensorflow as tf
import os
import numpy as np
from code.model import model_fn, params
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tensorflow.python import debug as tf_debug # for debugging
import glob
from PIL import Image
from tkinter import filedialog
import tkinter as tk
import sys
import time
#from TF_stixels.data.folder2TFRec2_BW import Frame2StxTfrecords  # NEW: use the special version to extract frame id's rather than labels!!!!!!
from data.folder2TFRec2 import Frame2StxTfrecords  # NEW: use the special version to extract frame id's rather than labels!!!!!!
import csv

# params
#H = 370
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

class image_predictor:
    def __init__(self, image_in, out_folder, model_dir, debug_image, show_images):

        # load the correct model
        if os.path.exists(model_dir + '/model_for_CRF.py'):
            from model_for_CRF import model_fn, params
            print('impotrting model function')
        else:
            print('No model file within directory - exiting!!!!')

        # Init class internal params - folders
        self.out_folder = out_folder                    # init output folder
        self.model_name = os.path.basename(model_dir)   # init model name
        self.model_dir = model_dir                      # init the model folder
        self.image_folder = os.path.dirname(out_folder) # init input image folder
        self.model_fn = model_fn
        self.params = params

        # Init class internal params - dimensions
        im = Image.open(image_in)
        self.image_size = im.size
        self.W = params.image_width                     # init stixel width
        self.H = params.image_height                    # init stixel height
        self.stixel_stride = 5

        stixels_num_float = (self.image_size[0] - self.W)/self.stixel_stride
        uncovered_image_pix = (stixels_num_float - int(stixels_num_float)) * self.stixel_stride
        self.image_width = int(self.image_size[0] - uncovered_image_pix)
        # print('image width = {}, stixel_width = {}, active image width = {}'.format(im.size[0], self.W, self.image_width))
        if (self.H == 370):
            self.prediction_to_pixels = 5
        elif (self.H==222):
            self.prediction_to_pixels = 3
        elif (self.H == 146):
            self.prediction_to_pixels = 2
        else:
            print('H unrecognized!!')
        self.y_spacing = int((self.prediction_to_pixels-1) / 2)
        print('y spacing = {}'.format(self.y_spacing))

        # Init class internal params - annotations
        self.plot_border_width = 2.0
        self.debug_image = debug_image
        self.show_images = show_images

        # Reset tf graph
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)  # possible values - DEBUG / INFO / WARN / ERROR / FATAL

        # Prepare data structure for softmax predictions
        self.grid_x_width = int((self.image_width - self.W) / self.stixel_stride) + 1
        #self.grid_x_width = int((self.image_width - 36) / 5) + 1
        self.grid_y_width = 74

        # Create data grid
        x = np.linspace(0, self.grid_x_width - 1, self.grid_x_width) * 5 + int(self.W/2) - int((self.stixel_stride+1)/2)  # reduce 3 to center the probability points
        #x = np.linspace(0, self.grid_x_width - 1, self.grid_x_width) * 5 + int(self.W/2) - 3
        y = np.linspace(0, self.grid_y_width - 1, self.grid_y_width) * self.prediction_to_pixels - self.y_spacing
        #y = np.linspace(0, self.grid_y_width - 1, self.grid_y_width) * 5 - 2 # For 370 height stixels
        self.X, self.Y = np.meshgrid(x, y)
        #print(np.shape(self.X), np.shape(self.Y))

        # Create and run as tensorflow session
        self.sess =  tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # Load the model
        self.estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

        # MAKE SURE TO CLOSE THE SESSION ONCE DONE !!!!

    def close_session(self):
        print('Closing the session')
        self.sess.close()

    #############################################################
    ###   Visualizing predictions and creating output video   ###
    #############################################################

    def visualize_pred(self, image_in, tfrecord_file, predictions_list):

        # Pipeline of dataset and iterator
        dataset = tf.data.TFRecordDataset([tfrecord_file])
        dataset = dataset.map(parse)
        iterator = dataset.make_one_shot_iterator()
        next_image_data = iterator.get_next()

        num_of_stixels = len(predictions_list)
        print('Number of stixels to be proceesed  {}'.format(num_of_stixels))

        # Go through the TFRecord and reconstruct the images + predictions

        # Init new image
        new_im = Image.new('RGB', (self.image_width, self.H))
        grid = np.zeros((self.grid_y_width, self.grid_x_width))
        x_offset = 0
        first_time = True
        fig, ax = plt.subplots()

        # Go through all the stixels in the tfrecord file
        #for i in range(num_of_stixels):
        for i in range(num_of_stixels):
            image_data = self.sess.run(next_image_data)
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
            if self.debug_image:
                plt.plot(int(params.image_width / 2) + 5 * (frame_id), prediction * self.prediction_to_pixels, marker='o', markersize=4, color="red")
            # visualize probabilities
            grid[:,frame_id] = prediction_softmax
            plt.draw()

        # Use CRF to find the best path
        from code.crf import viterbi
        best_path = viterbi(grid.T, N, T, W_trans)

        # Plot the CRF border line
        best_path_points = []
        for index, path in enumerate(best_path):
            best_path_points.append([int(params.image_width / 2) + index * 5, path * 5 + self.y_spacing])
        plt.plot(np.array(best_path_points)[:,0], np.array(best_path_points)[:,1], color="blue", linewidth=self.plot_border_width)

        # If labeles exist and not in debug mode, plot the labels
        annotation_in = image_in.replace('.jpg', '.csv')
        if os.path.exists(annotation_in):
            del_y = self.image_size[1] - self.H
            #del_y = self.image_size[1] - 370
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

            # Compute the prediction accuracy


            # If not in debug mode, display the labels
            if not self.debug_image:
                plt.plot(np.array(label_coords)[:, 0], np.array(label_coords)[:, 1], color="red", linewidth=1.0)



        if self.debug_image:
            #In debug mode plot the softmax probabilities
            grid = np.ma.masked_array(grid, grid < .0001)
            plt.pcolormesh(self.X, self.Y, grid, norm=colors.LogNorm(), alpha = 0.75)
        plt.imshow(new_im, cmap='gray', alpha=1.0, interpolation='none')

        name = ' {} N{}_T{}_Tr{}.jpg'.format(self.model_name, N, T, W_trans)
        if self.debug_image:
            name.replace('.jpg',' debug.jpg')
            name = ' {} N{}_T{}_Tr{} debug.jpg'.format(self.model_name, N, T, W_trans)
            print('replacing name to indicate debug !!!!!!')
            print(name)

        image_out_name = os.path.basename(image_in)
        image_out_name = image_out_name.replace('.jpg', name)
        image_out_name = os.path.basename(image_out_name)
        image_file_name = os.path.join(self.out_folder, image_out_name)
        plt.savefig(image_file_name, format='jpg')
        print('saving fig to ', image_file_name)

        if self.show_images:
            plt.show()
        plt.close()


    ######################################
    ### translate image ro a TFRecord ###
    #####################################

    def image_2_tfrec(self, image_in, tfrec_filename, model_stixel_width, model_stixel_height):

        start_time = time.time()
        # Create TFRec writer
        os.chdir(os.path.dirname(image_in))

        print('output file = ', tfrec_filename)
        writer = tf.python_io.TFRecordWriter(tfrec_filename)
        # parse the image and save it to TFrecord
        f_to_stx = Frame2StxTfrecords(image_in, image_in.replace('.jpg', '.csv'), writer, os.path.dirname(image_in),
                                      model_stixel_width, 'test', model_stixel_height)
        f_to_stx.create_stx(False)
        writer.close()
        duration = time.time() - start_time
        print('TFRec creation took {} sec'.format(int(duration)))

        '''
        if not (glob.glob(tfrec_filename)):
            print('output file = ', tfrec_filename)
            writer = tf.python_io.TFRecordWriter(tfrec_filename)
            # parse the image and save it to TFrecord
            f_to_stx = Frame2StxTfrecords(image_in, image_in.replace('.jpg','.csv'), writer, os.path.dirname(image_in), model_stixel_width, 'test', model_stixel_height)
            f_to_stx.create_stx(False)
            writer.close()
            duration = time.time() - start_time
            print('TFRec creation took {} sec'.format(int(duration)))
        else:
            print('tfrecord already exists - skipping TFrec creation!!')
            '''

    ######################################
    ###      start a new prediction   ###
    #####################################

    def predict(self, image_in):

        # Translate the image to a TF record
        tfrec_filename = image_in.replace('.jpg', '_W' + str(self.W) + '.tfrecord')
        self.image_2_tfrec(image_in, tfrec_filename, self.W, self.H)
        data_files = {'test': tfrec_filename}
        predictions = self.estimator.predict(input_fn=lambda: dataset_input_fn('test', data_files))

        # Predict!
        predictions_list = list(predictions)

        # Visualize predictions based on single test TFrecord
        self.visualize_pred(image_in, tfrec_filename, predictions_list)


if __name__ == '__main__':

    from tkinter.filedialog import askopenfilename

    # Determine input image
    image_in = '/media/dnn/ML/Results/image_for_predict/12_02_2019_10_02_56_63.jpg'
    #image_in = '/media/dnn/ML/Results/image_for_predict/frame_000142.jpg'
    if not os.path.exists(image_in):
        print('no such file')

    '''
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    # image_filename = askopenfilename(initialdir='/media/vision/In Process/ForAnnotation/GC23_fixes')  # show an "Open" dialog box and return the path to the selected file
    image_in = askopenfilename(
        initialdir='/media/vision/Results/',
        filetypes=[('JPG file', '*.jpg')])  # show an "Open" dialog box and return the path to the selected file
    root.destroy()
    print('predict image - ' + image_in)
    '''

    image_out_dir = os.path.dirname(image_in)

    # Determine the model
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-28_18-57-33_EP_250' # Last RGB training
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-04-23_12-04-16_EP_100'
    model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-04-24_12-10-14_EP_100'
    model_name = os.path.basename(model_dir)

    os.chdir(model_dir)
    sys.path.insert(0, os.getcwd())
    if os.path.exists(model_dir + '/model_for_CRF.py'):
        from model_for_CRF import model_fn, params
    else:
        print('No model file within directory - exiting!!!!')
        os.exit()

    # If required, create required folder
    image_out_dir = image_out_dir + '/' + model_name
    if not os.path.exists(image_out_dir) and not os.path.isdir(image_out_dir):
        os.mkdir(image_out_dir)

    # Create image_predictor object
    predictor = image_predictor(image_in, image_out_dir, model_dir, debug_image=True, show_images=True)
    predictor.predict(image_in)

    #image_in = '/media/dnn/ML/Results/image_for_predict/frame_000136.jpg'
    #predictor.predict(image_in)

    # Close the session
    predictor.close_session()









