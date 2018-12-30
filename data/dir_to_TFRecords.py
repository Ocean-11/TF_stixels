'''
*
* dir_to_tfrecords
*
* Purpose: the module receives a directory name, creates train/valid/test/control/meta_data
*          folders, and scans it's "annotated" folder, translating annotated images into
*          train/valid/test TFrecords. Control images and meta_data are also saved
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


class Frame2StxTfrecords:
    def __init__(self, frame_path, GT_file_path, writer,control_dir, stixel_width, frame_type):
        ' Define stixels dimensions'
        self.stx_w = stixel_width  # stixel width
        #self.stx_w = 24  # stixel width
        self.stx_half_w = int(self.stx_w/2)
        #print('stixel half width = {}'.format(self.stx_half_w))
        self.stx_h = 370  # stixel height
        self.bin_pixels = 5  # stixel bins height
        self.stride = 5  # stride used to split to the frame
        ' init internal data'
        self.frame_path = frame_path  # frame path
        self.GT_file_path = GT_file_path  # ground truth
        self.writer = writer
        self.labels = []
        self.control_dir = control_dir
        self.frame_type = frame_type

        ' Read GT file (retain only first 2 columns) '
        GT_data = pd.read_csv(GT_file_path, header=None)
        GT_df = pd.DataFrame(GT_data)
        GT_df = GT_df.iloc[:, 0:2] #getting rid of columns 3 & 4
        GT_df.columns = ["x","y"] # adding columns names
        self.frame_ground_truth = GT_df

        #self.frame_ground_truth = pd.read_csv(GT_file_path, names=["x", "y"])


    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

    def create_tf_example(self, img_raw, label):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': self.bytes_feature(img_raw),
            'label': self.int64_feature(label),
        }))
        return tf_example

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
        h, w, c = img.shape
        #print('image dimensions: h={} w={} c={}'.format(h, w, c))
        x_start = self.frame_ground_truth.x.iat[0]
        x_stop = self.frame_ground_truth.x.iat[-1]
        annotated_w = x_stop - x_start + 1
        #print('start={} stop={} w={}'.format(x_start, x_stop, annotated_w))
        num_stixels = int(((annotated_w - self.stx_w) / self.stride) + 1)
        print('number of stixles to be generated {}'.format(num_stixels))
        frame_name = os.path.basename(self.frame_path)

        ' display the current frame'
        fig, ax = plt.subplots()
        ax.imshow(img)

        for stixel in range(num_stixels):

            imName = '_'.join([os.path.splitext(frame_name)[0], 'stx', str(stixel).zfill(3)]) # RAN - 13-12
            #print(imName)
            #imName = '_'.join([self.output_prefix, os.path.splitext(frame_name)[0], 'stx', str(stixel).zfill(3)])

            i = self.stx_half_w + (stixel * 5) + x_start
            #i = 12 + (stixel * 5) + x_start
            # print('\nstixel {} center = {}'.format(stixel, i))q
            ' cut the lower image part (high y values)'
            if img.shape[0] == self.stx_h:
                s = img[:, i - self.stx_half_w:i + self.stx_half_w, :]  # that's the stixel
                #s = img[:, i - 12:i + 12, :]  # that's the stixel
            else:
                diff_h = img.shape[0] - self.stx_h
                s = img[diff_h:, i - self.stx_half_w:i + self.stx_half_w, :]  # that's the stixel
                #s = img[diff_h:, i - 12:i + 12, :]  # that's the stixel

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
            # print('Ground truth result = {}'.format(stx_GT_y))

            'translate from absolute GT coordinates to stixel coordinates and translate to label'
            stx_GT_y -= diff_h
            if (stx_GT_y <= 0):
                'no border in stixel range'
                stx_GT_y = 0
                for_use = 0
            else:
                for_use = 1
            stx_label = np.floor_divide(stx_GT_y, 5)
            #print('stixel GT value = {}, use={}'.format(stx_label,for_use))

            self.labels.append([imName, stx_label, for_use, self.frame_type])

            ' display the GT on the stixel or on the original image '
            # self.plot_stx_GT(imName,s,stx_label)
            if stx_label == 0:
                plt.plot(x_start + i, diff_h + (stx_label * 5) + 2, marker='o', markersize=2, color="black")
            else:
                plt.plot(x_start + i, diff_h + (stx_label * 5) + 2, marker='o', markersize=2, color="red")
            plt.draw()

            ' save a tfrecord file'
            img_for_tfrec = Image.fromarray(s)
            with io.BytesIO() as output:
                img_for_tfrec.save(output, format="PNG")
                contents = output.getvalue()
            tf_example = self.create_tf_example(contents, stx_label)


            '''
            ' save the stixel image '
            output_path = os.path.join(self.output_dir, imName + '_L' + (str(stx_label)).zfill(2) + '.png')
            im = Image.fromarray(s)   
            im.save(output_path)
            '''

            self.writer.write(tf_example.SerializeToString())
        # print('labels are: {}'.format(list(zip(*self.labels))[1]))

        if (printCtrlImage):
            ' print control image to script directory '

            control_imagepath = (self.control_dir + '/'+ os.path.basename(self.frame_path)).replace('.png', '_labeled.png')
            control_imagepath = control_imagepath.replace('.jpg', '_labeled.png')
            print('stixels created - saving control image to ' + control_imagepath)
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
    if not os.path.exists(data_dir + '/train') and not os.path.isdir(data_dir + '/train'):
        os.mkdir(data_dir + '/train')

    if not os.path.exists(data_dir + '/valid') and not os.path.isdir(data_dir + '/valid'):
        os.mkdir(data_dir + '/valid')

    if not os.path.exists(data_dir + '/test') and not os.path.isdir(data_dir + '/test'):
        os.mkdir(data_dir + '/test')

    if not os.path.exists(data_dir   + '/control') and not os.path.isdir(data_dir + '/control'):
        os.mkdir(data_dir + '/control')

    if not os.path.exists(data_dir + '/meta_data') and not os.path.isdir(data_dir + '/meta_data'):
        os.mkdir(data_dir + '/meta_data')

    # Create object labels list
    object_labels = []


    # Create TFRecord writers
    train_writer = tf.python_io.TFRecordWriter(data_dir + '/train/' + object_name + '_W' + str(stixel_width) + '.tfrecord')
    valid_writer = tf.python_io.TFRecordWriter(data_dir + '/valid/' + object_name + '_W' + str(stixel_width) + '.tfrecord')
    print('Opening writers for ' + object_name + ' train/valid records')
    #object_stixels_list = []

    # Create this object's images control directory
    control_dir = data_dir + '/control'
    '''
    control_dir = data_dir + '/tfrecords/control/' + object
    if isControl:
        if not os.path.exists(control_dir) and not os.path.isdir(control_dir):
            os.mkdir(control_dir)
            print('created images annotations control dir - ' + control_dir)
            '''


    # Go through object images
    for index in range(num_images):
        i = objects[index]

        # Make sure there's an image - otherwise move to next image
        frame_path = (i.split('.')[0]) + '.jpg'
        if os.path.exists(frame_path):
            print('parse: ' + frame_path)
        else:
            print(frame_path + ' does not exist!')
            continue

        # Let's make 80% train, 15% for validation & 5% for test
        isTestImage = False
        frame_type = 'train'
        if index < num_images * 0.8:
            writer = train_writer
            #print('write to test writer')
        elif index < num_images * 0.95:
            writer = valid_writer
            frame_type = 'valid'
            #print('write to valid writer')
        else:
            # Create new test writer for each image
            frame_name = os.path.basename(i.split('.')[0])
            test_writer = tf.python_io.TFRecordWriter(data_dir + '/test/' + object_name + ' ' + frame_name + '_test_W' + str(stixel_width) + '.tfrecord')
            writer = test_writer
            isTestImage = True
            frame_type = 'test'


        # parse the image and save it to TFrecord
        f_to_stx = Frame2StxTfrecords(frame_path, i, writer,control_dir, stixel_width, frame_type)
        frame_labels = f_to_stx.create_stx(isControl)
        object_labels.extend(frame_labels) # append the new frame labels data
        if isTestImage:
            # Close the test image writer
            test_writer.close()
            isTestImage = False

        # Save the 1st train image to test directory for further validation
        if index == 0:
            # Create new test writer
            frame_name = os.path.basename(i.split('.')[0])
            test_writer = tf.python_io.TFRecordWriter(
                data_dir + '/test/train_img_' +
                object_name + ' ' + frame_name + '_test_W' +
                str(stixel_width) + '.tfrecord')
            f_to_stx = Frame2StxTfrecords(frame_path, i, test_writer, control_dir, stixel_width, frame_type)
            f_to_stx.create_stx(isControl)
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

    print('Convert to TFrecords all annotated images within - ' + data_dir + ':')

    main(data_dir, params.image_width, True) # True saves a control image to a control directory
