

'''
*
* correct_annotation()
*
* Purpose: scan a directory with ill-annotated images, find the original annotation files
*          for each image in vision/DataRepo folder and initiate manual annotation
*
* Inputs:
*   faulty images folder
*
* Outputs:
*   fixed annotation file replacing the previous one
*
* Written by: Ran 09-01-2019
*
'''

import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
import csv
import sys
import re
import shutil
from show_image_annotations import show_image
from annotate_new_image import annotate_image


def main(image_filename):

    # Parse the prediction outcome filename and find the relevant image + annotations files
    folder_name = os.path.basename(image_filename.split(' ')[0])
    image_name = image_filename.split(' ')[-1]
    image_name = image_name.split('__')[0]
    image_name = re.sub(r'_test_W\d\d', '', image_name)
    image_name = image_name.replace('.tfrecord', '.jpg')
    image_path = os.path.join('/media/vision/DataRepo/'+folder_name+'/annotated', image_name)
    annotation_path = image_path.replace('.jpg','.csv')
    print('correct ' + image_path + ' annotations')

    # Call show image to display the image + annotations
    if not(os.path.isfile(image_path)) or not(os.path.isfile(annotation_path)):
        print('files do not exist')
        return

    # Show the annotated image & Prompt the user if a fix is needed
    '''
    show_image(image_path)    
    is_correct = input('Correct annotations? (y=yes, n=no): ')
    if is_correct != 'y':
        return
    '''

    # Store CSV backup, copy the image to pre-annotations folder and call annotate_new_image
    print('correcting ..')
    shutil.move(annotation_path, annotation_path.replace('.csv', '.csv_bu'))
    shutil.move(image_path, '/media/vision/DataRepo/'+folder_name)
    new_image_path = os.path.join('/media/vision/DataRepo/'+folder_name, image_name)

    # Call annotate_new_image()
    annotate_image(new_image_path)


if __name__ == '__main__':

    ' Choose a jpg image to show '
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    image_filename = askopenfilename(initialdir='/media/vision/Datasets')  # show an "Open" dialog box and return the path to the selected file
    print('image file - ' + image_filename)
    root.destroy() # destroy the root window at the end of openFile to allow the script to close

    #image_filename = 'NE1_Garden8 frame_000194_test_W36___Model_2019-01-08_23-32-03_EP_250.jpg'

    main(image_filename)