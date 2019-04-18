
'''
*
* crop_image()
*
* Purpose: crops an images folder using prescribed x limits
*
* Output: Cropped directory containing the cropped images
*
* Written by: Ran Zaslavsky 04-12-2019
'''


# imports
import glob, os
import matplotlib.image as mpimg
import tkinter as tk
from tkinter import filedialog
import shutil
import cv2


def main(image_name):
    rgb_file_name = image_name.replace('.jpg','_BW_RGB_format.jpg')
    image = cv2.imread(image_name,  cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('Grey Scale Image', image)
    #cv2.waitKey()

    backtorgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #cv2.imshow('GRAY2RGB Image', backtorgb)
    #cv2.waitKey()
    cv2.imwrite(rgb_file_name, backtorgb)



if __name__ == '__main__':

    ' when executed as a script, open a GUI window to select the presented TFrecord file '
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    image_name = filedialog.askopenfilename(initialdir='/media/vision/Results/')
    root.destroy()

    #print('Crop images within - ' + data_dir + ':')

    main(image_name) # True saves a control image to a control directory
