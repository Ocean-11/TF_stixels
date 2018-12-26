'''
*
* annotate_new_image()
*
* Purpose: the script selects an image, provides GUI for annotation, and once done
*          creates an annotation CSV file and copies both to the "annotated" folder
*
* Written by: Ran Zaslavsky Sat Sep 29 00:29:34 2018
*
'''

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import get_backend
import numpy as np
# from AnacondaProjects.Utils.extract_border_coords import ExtractCoords #
from extract_border_coords import ExtractCoords
import tkinter as tk
from tkinter.filedialog import askopenfilename

root = tk.Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
image_filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
print('annotate image - ' + image_filename)

if os.path.isfile(image_filename):

    img = mpimg.imread(image_filename)
    # imgplot = plt.imshow(img)
    fig, ax = plt.subplots()

    'maximize the window used'
    print('matplotlib backend: ' + get_backend())
    figManager = plt.get_current_fig_manager()
    figManager.window.state('withdrawn')  # maximize for TkAgg backend
    # figManager.window.showMaximized() # maximize for spider backend

    ax.imshow(img)

    border_extractor = ExtractCoords(fig, image_filename)
    # border_extractor = ExtractCoords(fig,image_filename,annotations_filename)
    border_extractor.connect()

else:
    print('not an image file')





