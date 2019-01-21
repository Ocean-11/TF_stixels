

#from __future__ import absolute_import

import tensorflow as tf
import os
import numpy as np
from TF_stixels.code.model import model_fn, params
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug # for debugging
import glob
from PIL import Image
from tkinter import filedialog
import tkinter as tk
import sys
import time
from TF_stixels.data.folder2TFRec2 import Frame2StxTfrecords # RAN (15-01)- changed from folder2TFRec() to add frame names to TFRec
#from TF_stixels.data.folder2TFRec import Frame2StxTfrecords

import cv2
import io


# params
H = 370
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



###################################################
### Scan images in folder and create a TFRecord ###
###################################################

def create_tfrec(video_in_folder, tfrec_filename, model_stixel_width):

    start_time = time.time()

    video_files = sorted(glob.glob(video_in_folder + '/*.jpg'))
    num_images = len(video_files)
    os.chdir(video_in_folder)

    # In case no there are no annotations, return
    if num_images == 0:
        print('no images within ' + video_in_folder)
        return
    else:
        print('Folder name: ' + video_in_folder + ' images to be converted - {}'.format(num_images))

    # Create TFRec writer

    if not (glob.glob('*.tfrecord')):
        print('output file = ', tfrec_filename)
        writer = tf.python_io.TFRecordWriter(tfrec_filename)

        # Go through object images and create the tfrecord
        # for index in range(num_images):
        for index in range(num_images):
            i = video_files[index]
            print(i)
            # parse the image and save it to TFrecord
            f_to_stx = Frame2StxTfrecords(i, i, writer, video_in_folder, model_stixel_width, 'test')
            f_to_stx.create_stx(False)

        writer.close()

        duration = time.time() - start_time
        print('TFRec creation took {} sec'.format(int(duration)))

    else:
        print('tfrecord already exists - skipping TFrec creation!!')


#############################################################
###   Visualizing predictions and creating output video   ###
#############################################################

def visualize_pred(tfrecord_file, predictions_list, image_width, video_out_name):

    #image_folder = '/media/vision/Results/For_video3/'
    image_folder = os.path.dirname(video_out_name)
    image_num = 0
    print(image_folder)

    # reset tf graph
    tf.reset_default_graph()

    # Pipeline of dataset and iterator
    dataset = tf.data.TFRecordDataset([tfrecord_file])
    dataset = dataset.map(parse)
    iterator = dataset.make_one_shot_iterator()
    next_image_data = iterator.get_next()

    num_of_stixels = len(predictions_list)
    print(num_of_stixels)

    # Go through the TFRecord and reconstruct the images + predictions
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Init new image
        new_im = Image.new('RGB', (image_width, 370))
        x_offset = 0
        first_time = True
        fig, ax = plt.subplots()

        # Go through all the stixels in the tfrecord file
        for i in range(num_of_stixels):

            image_data = sess.run(next_image_data)
            image = image_data[0]['image']
            im = Image.fromarray(np.uint8(image))
            frame_id = image_data[1]
            prediction = predictions_list[i]
            #print('frame id {} - prediction = {}'.format(frame_id,prediction))
            
            #label = image_data[1]
            #frame_id = image_data[2]
            #frame_name = image_data[3]
            
            # Start a new image when frame_id = 0
            if (frame_id == 0):
                if (first_time):
                    # do nothing and set first_time to False
                    first_time = False
                else:
                    # Display the image and init a new one
                    final_image = np.array(new_im)
                    ax.imshow(final_image)
                    image_file_name = os.path.join(image_folder, 'frame' + format(image_num, '03d') + '.jpg')
                    image_num = image_num + 1
                    plt.savefig(image_file_name)
                    print('saving fig to ', image_file_name)
                    plt.close()

                    # FUTURE: append the matplotlib image directly to video writer


                    #image_for_save = cv2.imread('/home/dev/PycharmProjects/stixel/TF_stixels/data/Dataset_12/For_video_GC23_2/frame_001767.jpg')
                    #video.write(image_for_save)
                    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    
                    #buf = io.BytesIO()
                    #plt.savefig(buf, format='png')
                    #buf.seek(0)
                    #im_for_video = Image.open(buf)
                    #buf.close()
                    
                    #plt.show()
                    #plt.close()
                    new_im = Image.new('RGB', (image_width, 370))
                    x_offset = 0
                    fig, ax = plt.subplots()

            # Collect and visualize stixels
            new_im.paste(im, (frame_id * 5, 0))
            x_offset += 5
            plt.plot(int(params.image_width / 2) + 5 * (frame_id), prediction * 5, marker='o', markersize=4, color="red")
            plt.draw()

        # Create the video
        #image_folder = '/media/vision/Results/For_video3/'
        images = [img for img in os.listdir((image_folder)) if img.endswith('.jpg')]
        frame_name = os.path.join(image_folder, images[0])
        print(frame_name)
        frame = cv2.imread(frame_name)
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(video_out_name, fourcc, 2, (width, height))
        # video = cv2.VideoWriter(video_out_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()


def main(video_in_folder, model_dir, model_stixel_width, image_width, video_out_name):

    # Scan images in folder and create a TFRecord
    tfrec_filename = os.path.join(video_in_folder, 'video_W' + str(model_stixel_width) + '.tfrecord')
    create_tfrec(video_in_folder, tfrec_filename, model_stixel_width)

    ##############################################################
    ### Create predictions for the Stixels in the video TFRec  ###
    ##############################################################

    # RAN - Setup logger - only displays the most important warnings
    tf.logging.set_verbosity(tf.logging.INFO)  # possible values - DEBUG / INFO / WARN / ERROR / FATAL

    # Load the model
    estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

    # Create new directory to save prediction annotated images
    model_dirname = os.path.basename(model_dir)

    # Create required folders
    '''
    if not os.path.exists(video_in_folder + '/' + model_dirname) and not os.path.isdir(video_in_folder + '/' + model_dirname):
        os.mkdir(video_in_folder + '/' + model_dirname)
        '''

    # locate the files to be processed
    test_file = tfrec_filename

    # Make sure test file stixels width are compatible with the model stixels width
    test_file_name = test_file.split('/')[-1]
    test_file_name = test_file_name.split(('.')[0])
    test_file_name = test_file_name[0]
    test_file_stixel_width = int(test_file_name.split('W')[-1])

    if test_file_stixel_width != model_stixel_width:
        print('Skipping ' + test_file + ' - does not match the model stixels width')
        return
    else:
        print('Process ' + test_file)

    # Now make all these files accessible depending on whether we
    #    are in training ('train') or validation ('valid') mode.
    data_files = {'test' : test_file}

    # Prepare hooks for debugging
    hooks = [tf_debug.TensorBoardDebugHook(grpc_debug_server_addresses="dev:6064")]

    start_time = time.time()

    predictions = estimator.predict(input_fn=lambda: dataset_input_fn('test',data_files))
    #predictions = estimator.predict(input_fn=lambda: dataset_input_fn('test'), hooks = hooks)

    # Print prediction duration
    duration = time.time() - start_time
    print('prediction took {}ms'.format(duration*1000))

    # Predict!
    predictions_list = list(predictions)

    # Visualize predictions based on single test TFrecord
    visualize_pred(test_file, predictions_list, image_width, video_out_name)

    # Print tensorboard data
    print('tensorboard --logdir=' + str(model_dir) + '--port 6006')
    #print('tensorboard --logdir=' + str(model_dir) + '--port 6006 --debugger_port 6064')



if __name__ == '__main__':

    #in_folder_name = 'test_video_GC23_1'
    #in_folder_name = 'test_video_GC23_2'
    #in_folder_name = 'test_video_NLSite_1'
    in_folder_name = 'test_video_Site40_1'

    # Determine the model to be used for inference
    #model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-10_19-59-23_EP_250'

    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    model_dir = filedialog.askdirectory(initialdir='/home/dev/PycharmProjects/stixel/TF_stixels/results')
    root.destroy()

    model_name = os.path.basename(model_dir)

    # Create Output directory
    video_in_folder = '/media/vision/Results/' + in_folder_name
    video_out_folder = video_in_folder + '/' + model_name
    video_out_name = os.path.join(video_out_folder, in_folder_name + '.avi')
    print(video_in_folder)
    print(video_out_name)
    if not os.path.exists(video_out_folder) and not os.path.isdir(video_out_folder):
        os.mkdir(video_out_folder)

    #video_in_folder = '/home/dev/PycharmProjects/stixel/TF_stixels/data/Dataset_12/For_video_GC23_2'
    #video_out_name = '/media/vision/Results/video.avi'

    # Restore model.py that was saved in model directory
    os.chdir(model_dir)
    sys.path.insert(0, os.getcwd())
    from model import model_fn, params

    W = params.image_width
    image_width = 476 # when image width = 480
    #image_width = 636 # when image width = 636

    # Add automatic check .............

    # Make sure the chosen directory contains a valid model file before calling main()
    if os.path.exists(model_dir + '/model.py'):
        print('Model file exists')
        main(video_in_folder, model_dir, W, image_width, video_out_name)
    else:
        print('No model file within directory - exiting!!!!')





