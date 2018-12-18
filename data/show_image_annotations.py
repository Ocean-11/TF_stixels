
'''
*
* show_image_annotations()
*
* Purpose: the script selects an image, opens the accompanying CSV annotations
*          file and displays the annotations on top of the image
*
* Note: Annotation CSV file name should be identical to the chosen image
*       (except for the postscript)
*
* Written by: Ran Zaslavsky Sat Sep 29 00:29:34 2018
*
'''

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import get_backend
import tkinter as tk
from tkinter.filedialog import askopenfilename
import csv

root = tk.Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
image_filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
print('image file - ' + image_filename)

if os.path.isfile(image_filename):

    img = mpimg.imread(image_filename)
    fig, ax = plt.subplots()

    'maximize the window used'
    print('matplotlib backend: ' + get_backend())
    figManager = plt.get_current_fig_manager()
    figManager.window.state('iconic')  # maximize for TkAgg backend python 3.5
    #figManager.window.state('zoomed')  # maximize for TkAgg backend
    # figManager.window.showMaximized() # maximize for spider backend

    ax.imshow(img)

    'read the annotation file and draw the border lines'
    annotation_filename = image_filename.replace('.png', '.csv')
    annotation_filename = annotation_filename.replace('.jpg', '.csv')

    coords = []
    x = []
    y = []
    with open(annotation_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            new_tuple = tuple(row)
            # print(new_tuple)
            x_coord = int(new_tuple[0])
            y_coord = int(new_tuple[1])
            # print('x = {}, y = {}'.format(x_coord,y_coord))
            plt.plot(x_coord, y_coord, marker='o', markersize=4, color="red")
            coords.append([x_coord, y_coord])
            x.append(x_coord)
            y.append(y_coord)
    plt.draw()

    print(coords)
    plt.plot(x, y, 'b-')
    plt.show()
    print('image shown')

    '''
    plt.plot([x_coord], [y_coord], marker='o', markersize=3, color="red")
    'draw the border line'
    #plt.plot(*zip(*self.coords))
    border_line.set_data(*zip(*coords)) #update the border line with the new coordinate
    plt.draw()                          
    '''

else:
    print('not an image file')
