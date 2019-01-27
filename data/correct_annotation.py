

'''
*
* correct_annotation()
*
* Purpose: open a dialog box that enables selecting a labeled file for correction,
*          searches DataRepo for the same folder + image name and allows annotation
*          correction
*
* Inputs:
*   faulty image (stored within the folder with the exact same name as in DataRepo)
*
* Outputs:
*   fixed annotation file
*
* Written by: Ran 27-01-2019
*
'''

import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
import re
from extract_border_coords import annotate_image

def main(image_filename):

    # Parse the prediction outcome filename and find the relevant image + annotations files
    #folder_name = os.path.basename(image_filename.split(' ')[0])
    folder_name = os.path.basename(os.path.dirname(image_filename))
    image_name = os.path.basename(image_filename)
    image_name = re.sub('_labeled.png', '.jpg', image_name)

    '''
    image_name = image_name.split('__')[0]
    image_name = re.sub(r'_test_W\d\d', '', image_name)
    image_name = image_name.replace('.tfrecord', '.jpg')    
    print('folder name = {}'.format(folder_name))
    print("image name = {}".format(image_name))
    '''

    image_path = os.path.join('/media/vision/DataRepo/'+folder_name+'/annotated', image_name)
    print('correct: ' + image_path)

    # If annotation files exist call annotate_image
    if not(os.path.isfile(image_path)):
        print('files do not exist')
        return
    else:
        print('opening file')
        annotate_image(image_path)


if __name__ == '__main__':

    ' Choose a jpg image to show '
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    image_filename = askopenfilename(initialdir='/media/vision/In Process/For corrections')  # show an "Open" dialog box and return the path to the selected file
    print('image file - ' + image_filename)
    root.destroy() # destroy the root window at the end of openFile to allow the script to close

    #image_filename = 'NE1_Garden8 frame_000194_test_W36___Model_2019-01-08_23-32-03_EP_250.jpg'

    main(image_filename)