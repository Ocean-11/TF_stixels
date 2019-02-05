'''
*
* folder2TFRec()
*
* Purpose: the module receives a directory name, creates train/valid/test/control/meta_data
*          folders, and scans it's "annotated" folder, translating annotated images into
*          stixels TFrecords. Note that lowest bound limit stixels diluted by a factor of 2
*          to reduce classification bias
*
*
* Inputs:
*   frame_path - annotated image path
*
* Outputs:
*   Stixles tfrecord files are saved into train/valid/test folders
*   meta_data CSV file saved to meta_data folder
*
*  Method:
*   1) Gather all the data (e.g. a list of images and corresponding labels)
*   2) Create a TFRecordWriter
*   3) Create an Example out of Features for each datapoint
*   4) Serialize the Examples
*   5) Write the Examples to your TFRecord
*
* Conventions: (x=0, y=0) is the upper left corner of the image
*
* Written by: Ran Zaslavsky 10-12-2018
'''


# imports

import tensorflow as tf
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
from PIL import Image
import scipy.interpolate
import io
import tkinter as tk
from tkinter import filedialog
import csv
from TF_stixels.code.model import params
import random
import shutil

TRAIN_IMAGES_RATIO = 80
VALID_IMAGES_RATIO = 15

class Frame2StxTfrecords:
    def __init__(self, frame_path, GT_file_path, writer,control_dir, stixel_width, frame_type):
        ' Define stixels dimensions'
        self.stx_w = stixel_width  # stixel width
        self.stx_half_w = int(self.stx_w/2)
        self.stx_h = 370  # stixel height
        self.bin_pixels = 5  # stixel bins height
        self.stride = 5  # stride used to split to the frame
        ' init internal data '
        self.frame_path = frame_path  # frame path
        self.GT_file_path = GT_file_path  # ground truth
        self.writer = writer
        self.labels = []
        self.control_dir = control_dir
        self.frame_type = frame_type

        ' Read GT file (retain only first 2 columns) '
        if (GT_file_path.endswith('.csv')):
            GT_data = pd.read_csv(GT_file_path, header=None)
            self.is_GT = True
        else:
            GT_data = pd.read_csv('/home/dev/PycharmProjects/stixel/TF_stixels/default_GT.csv', header=None)
            self.is_GT = False
            print('no CSV file given - using default')
        GT_df = pd.DataFrame(GT_data)
        GT_df = GT_df.iloc[:, 0:2] #getting rid of columns 3 & 4
        GT_df.columns = ["x","y"] # adding columns names
        self.frame_ground_truth = GT_df

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

    def string_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.strip().encode("ascii")]))

    def create_tf_example(self, img_raw, label):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': self.bytes_feature(img_raw),
            'label': self.int64_feature(label),
        }))
        return tf_example

    ## NEW: supporting frame_id for CRF prediction - may not be used during training
    def create_tf_example_2(self, img_raw, label):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': self.bytes_feature(img_raw),
            'frame_id': self.int64_feature(label),
        }))
        return tf_example

    # NEW
    '''
    def create_tf_example_2(self, img_raw, label, name):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': self.bytes_feature(img_raw),
            'label': self.int64_feature(label),
            'frame_id': self.int64_feature(label),
            'name': self.string_feature(name),
        }))
        return tf_example
        '''

    def plot_stx_GT(self, imName, s, stx_label):
        fig, ax = plt.subplots()
        ax.imshow(s)
        plt.plot(self.stx_half_w, stx_label * 5, marker='o', markersize=4, color="red")
        #plt.plot(12, stx_label * 5, marker='o', markersize=4, color="red")
        plt.title(imName)
        plt.draw()


    def create_stx(self, printCtrlImage):
        ' read current frame '
        img = mpimg.imread(self.frame_path)  # img is np.array of the frame
        height, width, c = img.shape
        #print('image dimensions: h={} w={} c={}'.format(height, width, c))
        x_start = self.frame_ground_truth.x.iat[0]
        x_stop = min(self.frame_ground_truth.x.iat[-1], width)
        #x_stop = self.frame_ground_truth.x.iat[-1]
        annotated_w = x_stop - x_start + 1
        #print('start={} stop={} w={}'.format(x_start, x_stop, annotated_w))
        num_stixels = int(((annotated_w - self.stx_w) / self.stride) + 1)
        #print('stixel width = {}, number of stixles to be generated {}'.format(self.stx_w, num_stixels))
        frame_name = os.path.basename(self.frame_path)

        ' display the current frame'
        fig, ax = plt.subplots()
        ax.imshow(img)

        for stixel in range(num_stixels):

            imName = '_'.join([os.path.splitext(frame_name)[0], 'stx', str(stixel).zfill(3)]) # RAN - 13-12
            #print('image name = ' + imName)
            #imName = '_'.join([self.output_prefix, os.path.splitext(frame_name)[0], 'stx', str(stixel).zfill(3)])

            i = self.stx_half_w + (stixel * 5) + x_start
            #i = 12 + (stixel * 5) + x_start
            #print('\nstixel {} center = {}'.format(stixel, i))
            ' cut the lower image part (high y values)'
            if img.shape[0] == self.stx_h:
                s = img[:, i - self.stx_half_w:i + self.stx_half_w, :]  # that's the stixel
                print('diff_h not defined !!!!!!!!')
                #s = img[:, i - 12:i + 12, :]  # that's the stixel
            else:
                diff_h = img.shape[0] - self.stx_h
                s = img[diff_h:, i - self.stx_half_w:i + self.stx_half_w, :]  # that's the stixel
                # s = img[diff_h:, i - 12:i + 12, :]  # that's the stixel

            ' find the closest GT points '
            stx_GT_y = -1
            GT_point_ID_1 = next(x[0] for x in enumerate(self.frame_ground_truth.x) if x[1] > i)
            GT_point_ID_2 = max(GT_point_ID_1 - 1, 0)

            '''
            print('GT for stixel {} points - ({},{}), ({},{})'.format(
                stixel,
                self.frame_ground_truth.x[GT_point_ID_2],
                self.frame_ground_truth.y[GT_point_ID_2],
                self.frame_ground_truth.x[GT_point_ID_1],
                self.frame_ground_truth.y[GT_point_ID_1]))
                '''

            ' interpolate the GT value'
            x_GT = [self.frame_ground_truth.x[GT_point_ID_2], self.frame_ground_truth.x[GT_point_ID_1]]
            y_GT = [self.frame_ground_truth.y[GT_point_ID_2], self.frame_ground_truth.y[GT_point_ID_1]]
            y_interp = scipy.interpolate.interp1d(x_GT, y_GT)
            stx_GT_y = int(y_interp(i))
            #print('Ground truth result = {}'.format(stx_GT_y))

            'translate from absolute GT coordinates to stixel coordinates and translate to label'
            stx_GT_y -= diff_h
            for_use = 1 # use all Stixels
            if (stx_GT_y < 0):
                stx_GT_y = 0 # label as "0"

            stx_label = np.floor_divide(stx_GT_y, 5)
            #print('stixel GT value = {}, use={}'.format(stx_label,for_use))

            ' display the GT on the stixel or on the original image '
            # self.plot_stx_GT(imName,s,stx_label)
            if stx_label == 0:
                plt.plot(i, diff_h + (stx_label * 5) + 2, marker='o', markersize=2, color="black")
            else:
                plt.plot(i, diff_h + (stx_label * 5) + 2, marker='o', markersize=2, color="red")
            plt.draw()

            '''
            if stx_label == 0:
                plt.plot(x_start + i, diff_h + (stx_label * 5) + 2, marker='o', markersize=2, color="black")
            else:
                plt.plot(x_start + i, diff_h + (stx_label * 5) + 2, marker='o', markersize=2, color="red")
            plt.draw()
            '''

            ' dilute the amount of lower end border Stixels by a factor of 1:4'
            '''
            dilution_factor = 2
            if (stx_label == 73):
                p_label = random.randint(1,dilution_factor + 1)
                #print('prob = {}'.format(p_label))
                if (p_label>1):
                    for_use = 0
            '''

            ' save the stixel only if there is a boundary within the stixel'
            if for_use == 1:
                ' Apend the new stixel label'
                self.labels.append([imName, stx_label, for_use, self.frame_type])

                ' save a tfrecord file'
                img_for_tfrec = Image.fromarray(s)
                with io.BytesIO() as output:
                    img_for_tfrec.save(output, format="PNG")
                    contents = output.getvalue()
                #tf_example = self.create_tf_example(contents, stx_label)
                tf_example = self.create_tf_example_2(contents, stixel) ### NEW - save the frame_id, rather than stixel label
                self.writer.write(tf_example.SerializeToString())

        # print('labels are: {}'.format(list(zip(*self.labels))[1]))

        if (printCtrlImage):
            ' print control image to script directory '

            control_imagepath = (self.control_dir + '/'+ os.path.basename(self.frame_path)).replace('.png', '_labeled.png')
            control_imagepath = control_imagepath.replace('.jpg', '_labeled.png')
            #print('stixels created - saving control image to ' + control_imagepath)
            plt.savefig(control_imagepath)

        plt.close('all')

        return self.labels


def main(data_dir, stixel_width, isControl = True):

    # Gather annotated images paths
    object_name = os.path.basename(os.path.normpath(data_dir))
    object_dirs = data_dir + '/annotated'
    objects = glob.glob(object_dirs + '/*.csv')
    num_images = len(objects)

    # In case no there are no annotations, return
    if num_images == 0:
        print('no images within ' + object_name)
        return
    else:
        print('Folder name: ' + object_name + 'images to be converted - {}'.format(num_images))

    # Create required folders
    data_dir = data_dir+'/W'+ str(stixel_width)
    print(data_dir)
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    folders = {'train', 'valid', 'test', 'control', 'meta_data'}
    for folder_ in folders:
        new_folder = data_dir + '/' + folder_
        '''
        if os.path.exists(new_folder) and os.path.isdir(new_folder):
            shutil.rmtree(new_folder)
        '''
        os.mkdir(new_folder)
        print('create ' + new_folder)

    # Create object labels list
    object_labels = []

    # Create TFRecord writers
    train_writer = tf.python_io.TFRecordWriter(data_dir + '/train/' + object_name + '_W' + str(stixel_width) + '.tfrecord')
    valid_writer = tf.python_io.TFRecordWriter(data_dir + '/valid/' + object_name + '_W' + str(stixel_width) + '.tfrecord')
    print('Opening writers for ' + object_name + ' train/valid records')
    #object_stixels_list = []

    # Define images control directory
    control_dir = data_dir + '/control'
    capture_train_image = True

    # Go through object images
    for index in range(num_images):
        i = objects[index]

        # Make sure there's an image - otherwise move to next image
        frame_path = (i.split('.')[0]) + '.jpg'
        if not(os.path.exists(frame_path)):
            print(frame_path + ' does not exist!')
            continue

        # Split images between train/validation/test
        type_val = random.randint(1, 101)
        isTestImage = False
        if type_val<=TRAIN_IMAGES_RATIO:
            writer = train_writer
            frame_type = 'train'
            print('parse: ' + frame_path + '->train')
        elif type_val<=(TRAIN_IMAGES_RATIO+VALID_IMAGES_RATIO):
            writer = valid_writer
            frame_type = 'valid'
            print('parse: ' + frame_path + '->valid')
        else:
            # Create new test writer for each image
            frame_type = 'test'
            frame_name = os.path.basename(i.split('.')[0])
            test_writer = tf.python_io.TFRecordWriter(
                data_dir + '/test/' + object_name + ' ' + frame_name + '_test_W' + str(stixel_width) + '.tfrecord')
            writer = test_writer
            isTestImage = True
            print('parse: ' + frame_path + '->test')

        # parse the image and save it to TFrecord
        f_to_stx = Frame2StxTfrecords(frame_path, i, writer,control_dir, stixel_width, frame_type)
        frame_labels = f_to_stx.create_stx(isControl)
        object_labels.extend(frame_labels) # append the new frame labels data
        if isTestImage:
            # Close the test image writer
            test_writer.close()
            isTestImage = False

        # Save the 1st train image to test directory for further validation
        if (frame_type == 'train') and (capture_train_image):
            # Create new test writer
            frame_name = os.path.basename(i.split('.')[0])
            test_writer = tf.python_io.TFRecordWriter(
                data_dir + '/test/train_img_' +
                object_name + ' ' + frame_name + '_test_W' +
                str(stixel_width) + '.tfrecord')
            f_to_stx = Frame2StxTfrecords(frame_path, i, test_writer, control_dir, stixel_width, frame_type)
            f_to_stx.create_stx(isControl)
            capture_train_image = False
            test_writer.close()

    train_writer.close()
    valid_writer.close()
    #test_writer.close()

    'save coordinates to file'
    meta_data_dir = os.path.join(data_dir + '/meta_data')
    out_filename = os.path.join(meta_data_dir,object_name + '_W' + str(stixel_width) +  '.csv')

    with open(out_filename, 'w', newline='', encoding="utf-8") as f_output:
        csv_output = csv.writer(f_output)
        csv_output.writerows(object_labels)
        print('writing csv file to ' + out_filename)


if __name__ == '__main__':



    ' when executed as a script, open a GUI window to select the presented TFrecord file '
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    data_dir = filedialog.askdirectory(initialdir='/media/vision/DataRepo')
    root.destroy()

    print('Convert to TFrecords all annotated images within - ' + data_dir + ':')

    main(data_dir, params.image_width, True) # True saves a control image to a control directory
