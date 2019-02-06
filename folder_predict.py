

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
from TF_stixels.data.folder2TFRec2 import Frame2StxTfrecords # RAN (15-01)- changed from folder2TFRec() to add frame names to TFRec

import cv2
import io


# params
H = 370
C = 3

# Init CRF parameters
N = 1
T = 20 #20
W_trans = 10

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

############################################
###   Viterbi algorithm implementation   ###
############################################

'''
# from https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm:

def viterbi(self, observations):
    """Return the best path, given an HMM model and a sequence of observations"""
    # A - initialise stuff
    nSamples = len(observations[0])
    nStates = self.transition.shape[0]          # number of states
    c = np.zeros(nSamples)                      # scale factors (necessary to prevent underflow)
    viterbi = np.zeros((nStates, nSamples))     # initialise viterbi table
    psi = np.zeros((nStates, nSamples))         # initialise the best path table
    best_path = np.zeros(nSamples);             # this will be your output

    # B- appoint initial values for viterbi and best path (bp) tables - Eq (32a-32b)
    viterbi[:, 0] = self.priors.T * self.emission[:, observations(0)]
    c[0] = 1.0 / np.sum(viterbi[:, 0])
    viterbi[:, 0] = c[0] * viterbi[:, 0]  # apply the scaling factor
    psi[0] = 0;

    # C- Do the iterations for viterbi and psi for time>0 until T
    for t in range(1, nSamples):  # loop through time
        for s in range(0, nStates):  # loop through the states @(t-1)
            trans_p = viterbi[:, t - 1] * self.transition[:, s]
            psi[s, t], viterbi[s, t] = max(enumerate(trans_p), key=operator.itemgetter(1))
            viterbi[s, t] = viterbi[s, t] * self.emission[s, observations(t)]

        c[t] = 1.0 / np.sum(viterbi[:, t])  # scaling factor
        viterbi[:, t] = c[t] * viterbi[:, t]

    # D - Back-tracking
    best_path[nSamples - 1] = viterbi[:, nSamples - 1].argmax()  # last state
    for t in range(nSamples - 1, 0, -1):  # states of (last-1)th to 0th time step
        best_path[t - 1] = psi[best_path[t], t]

    return best_path
    '''

#############################################################################
### Creates a matrix weighting the transition between cell xi to cell xj) ###
#############################################################################

def create_transition_matrix(dim, N, T, Wb):

    transition_mat = np.zeros((dim,dim))
    for row in range(dim):
        for column in range (dim):
            transition_mat[column,row] = max ((abs(column-row)) - N, 0)
            transition_mat[column, row] = min(transition_mat[column,row], T)
            transition_mat[column, row] = transition_mat[column,row] * Wb

    return transition_mat


def viterbi(observations_matrix):

    observations_shape = np.shape(observations_matrix)

    # Go through the columns and compute following Viterbi function: Wu*P + Wb*min(max((|y1-y2|-N,0),T)
    N = 2
    T = 10
    W_unary = 1
    W_trans = 10
    stixel_dim = observations_shape[1]
    border_length = observations_shape[0]

    print('stixel_cells_num = {}'.format(stixel_dim))

    # init
    #viterbi = np.zeros((rows_num,1))
    viterbi = -np.log(observations_matrix[0,:])*W_unary
    #for cell_num, log_prob in enumerate(viterbi):
        #print('cell {} - log prob = {}'.format(cell_num, log_prob))

    transiotion_matrix = create_transition_matrix(stixel_dim,N,T, W_trans)
    transit_to_cell = 30
    '''
    for cell in range(dim):
        print('transition from cell {} to cell {} weight = {}'.format(cell, transit_to_cell, transiotion_matrix[cell,transit_to_cell]))
        '''
    # for transition in range(border_length - 1):
    for transition in range(1):
        new_row = transition + 1
        viterbi = viterbi + -np.log(observations_matrix[new_row,:])*W_unary

        #print(len(observations_matrix[new_line_num,:]))

        # Go through each column in the new line and calculate the new viterbi value based on previous line
        #for




        '''
        for row in range(rows_num):
            viterbi[row] = viterbi[row] + Wb * column[row]
            '''

    '''
    for row_id in range(rows_num):
        print('row {} sum: {}'.format(row_id,viterbi[row_id]))
        '''










    #for x in

    best_path = 0

    return best_path

#############################################################
###   Visualizing predictions and creating output video   ###
#############################################################

def visualize_pred(tfrecord_file, predictions_list, image_width, create_video, video_out_name):

    #image_folder = '/media/vision/Results/For_video3/'
    image_folder = os.path.dirname(video_out_name)
    image_num = 0
    print(image_folder)

    plot_border_width = 2.0

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
    #x = np.linspace(0, grid_x_width - 1, grid_x_width)*5 + 18
    #y = np.linspace(0, grid_y_width - 1, grid_y_width)*5
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
            #print('frame id {} - prediction = {}'.format(frame_id,prediction))

            #####################################################################################
            ## Collect all image predictions into a new array tp be filtered by CRF++/CRFSuite ##
            #####################################################################################

            #label = image_data[1]
            #frame_id = image_data[2]
            #frame_name = image_data[3]
            
            # Start a new image when frame_id = 0
            if (frame_id == 0):
                if (first_time):
                    # do nothing and set first_time to False
                    first_time = False
                else:
                    # Find the best path and display it (blue)
                    from code.crf import viterbi
                    best_path = viterbi(grid.T, N, T, W_trans)

                    # Plot the CRF border line
                    best_path_points = []
                    for index, path in enumerate(best_path):
                        best_path_points.append([int(params.image_width / 2) + index * 5, path * 5 + 2])
                    plt.plot(np.array(best_path_points)[:, 0], np.array(best_path_points)[:, 1], color="blue", linewidth=plot_border_width)

                    # pcolormesh of interpolated uniform grid with log colormap
                    if not(create_video):
                        grid = np.ma.masked_array(grid, grid < .00001)
                        plt.pcolormesh(X, Y, grid, norm=colors.LogNorm(), alpha=0.75)
                    plt.imshow(new_im, cmap='gray', alpha=1.0, interpolation='none')

                    image_file_name = os.path.join(image_folder, 'frame' + format(image_num, '03d') + '.jpg')
                    image_num = image_num + 1
                    plt.savefig(image_file_name, format='jpg')
                    #plt.savefig(image_file_name, format='jpg', dpi=600)
                    print('saving fig to ', image_file_name)
                    plt.close()

                    # FUTURE: append the matplotlib image directly to video writer
                    new_im = Image.new('RGB', (image_width, 370))
                    grid = np.zeros((grid_y_width, grid_x_width))
                    x_offset = 0
                    fig, ax = plt.subplots()

            # Collect and visualize stixels
            new_im.paste(im, (frame_id * 5, 0))
            x_offset += 5
            if not(create_video):
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


        grid = np.ma.masked_array(grid, grid < .01)
        # pcolormesh of interpolated uniform grid with log colormap
        plt.pcolormesh(X, Y, grid, norm=colors.LogNorm(), alpha = 0.75)
        plt.imshow(new_im, cmap='gray', alpha=1.0, interpolation='none')
        # Create the video or last image
        if create_video:
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
        else:
            plt.show()
            plt.close()



def main(video_in_folder, model_dir, model_stixel_width, image_width, create_video, video_out_name):

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
    visualize_pred(test_file, predictions_list, image_width, create_video, video_out_name)

    # Print tensorboard data
    print('tensorboard --logdir=' + str(model_dir) + '--port 6006')
    #print('tensorboard --logdir=' + str(model_dir) + '--port 6006 --debugger_port 6064')



if __name__ == '__main__':

    create_video = False

    image_width = 476  # when image width = 480
    #in_folder_name = '/media/vision/Results/image_for_predict'
    in_folder_name = '/media/vision/Results/test_video_GC23_1'
    #in_folder_name = 'test_video_GC23_2'
    #in_folder_name = 'test_video_Site40_1'
    #in_folder_name = 'test_video_single'
    #in_folder_name = 'test_video_NLSite_1'

    '''
    in_folder_name = 'test_video_UKSite4GC'
    image_width = 636  # when image width = 636
    '''
    '''
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    model_dir = filedialog.askdirectory(initialdir='/home/dev/PycharmProjects/stixel/TF_stixels/results')
    root.destroy()
    '''

    images = sorted(glob.glob(in_folder_name + '/*.jpg'))

    out_folder_name = in_folder_name

    # Determine the model to be used for inference
    model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-28_18-57-33_EP_250'
    model_name = os.path.basename(model_dir)


    import image_predict

    #image_in = '/media/vision/Results/image_for_predict/frame_000142.jpg'
    #out_folder_name = os.path.dirname(image_in)

    out_folder_name = in_folder_name

    for image in images:
        image_predict.main(image, out_folder_name, model_dir, image_width, False, show_images=False)



    '''
    
    # Create Output directory
    video_in_folder = '/media/vision/Results/' + in_folder_name
    video_out_folder = video_in_folder + '/' + model_name
    name = '{}_N{}_T{}_Tr{}.avi'.format(in_folder_name, N, T, W_trans)    #in_folder_name + '.avi'
    video_out_name = os.path.join(video_out_folder, name)
    print(video_in_folder)
    print(video_out_name)
    if not os.path.exists(video_out_folder) and not os.path.isdir(video_out_folder):
        os.mkdir(video_out_folder)

    #video_in_folder = '/home/dev/PycharmProjects/stixel/TF_stixels/data/Dataset_12/For_video_GC23_2'
    #video_out_name = '/media/vision/Results/video.avi'

    # Restore model.py that was saved in model directory
    os.chdir(model_dir)
    sys.path.insert(0, os.getcwd())
    if os.path.exists(model_dir + '/model_for_CRF.py'):
        from model_for_CRF import model_fn, params
    else:
        os.exit()

    W = params.image_width

    # Make sure the chosen directory contains a valid model file before calling main()
    if os.path.exists(model_dir + '/model_for_CRF.py'):
        print('Model file exists')
        main(video_in_folder, model_dir, W, image_width, create_video, video_out_name)
    else:
        print('No model file within directory - exiting!!!!')


    '''


