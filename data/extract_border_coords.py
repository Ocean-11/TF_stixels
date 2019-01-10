'''
*
* Created on Thu Oct  4 16:08:25 2018
*
* Purpose: Helper class for the annotate_new_image
*
* Inputs: active figure, image filename
*
* Outputs: annotation CSV file (original image & CSV are copied to /annotated dir
*
* User Interface:
* --------------
* left mouse button press = insert new border point
* right mouse button press on existing point = remove
* z key = remove right most point
* enter key  = save to csv file, move the original image to "annotated" directory and quit
*
* @author: Ran Zaslavsky
*
'''

import matplotlib.pyplot as plt
import csv
import shutil
import os
import matplotlib.image as mpimg
from matplotlib import get_backend
import tkinter as tk
from tkinter.filedialog import askopenfilename

' define key for coordinates sort '
def getKey(item):
    return(item[0])

class ExtractCoords:
    def __init__(self,my_figure,image_filename):

        # If annotation file already exists display it, otherwise prepare a new file
        image_name = os.path.basename(image_filename)
        annotation_name = image_name.replace('.png','.csv')
        annotation_name = annotation_name.replace('.jpg','.csv')

        # Initializing internal data
        self.coords = []  # list of coordinates tuple
        self.fig = my_figure
        self.axes = my_figure.axes
        self.image_file = image_filename
        self.output_dir = os.path.dirname(image_filename)
        self.out_filename = os.path.join(self.output_dir, annotation_name)
        self.point_ID = 0
        'prepare the border line'
        self.border_line, = plt.plot(*zip(*self.coords[0:1]), 'r-')
        print('init coordinates extractor')

        if os.path.exists(self.out_filename):
            # Init from the CSV file
            with open(self.out_filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for index, row in enumerate(csv_reader):
                    new_tuple = tuple(row)
                    x_coord = int(new_tuple[0])
                    y_coord = int(new_tuple[1])
                    border_point = plt.plot(x_coord, y_coord, marker='o', markersize=4, color="red", picker = 5)
                    self.coords.append(tuple((x_coord, y_coord, index, border_point)))
                # init point_ID
                self.point_ID = index + 1

                # Print the loaded coordinates
                for coord in self.coords:
                    print(coord)

            're-draw the border line'
            new_border_line = [(i[0], i[1]) for i in self.coords]  # get first 2 elemens in the tuple
            self.border_line.set_data(*zip(*new_border_line))  # update the border line with the new coordinate
            plt.draw()

            # update

        else:
            print('no annotations file found - prepare a new one')
            output_dir = os.path.dirname(image_filename) + '/annotated'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print('outpur dir created')
            annotations_filename = os.path.join(output_dir, annotation_name)
            self.output_dir = output_dir
            self.out_filename = annotations_filename
            print('output file - ', annotations_filename)



    def connect(self):
        'connect to all the events we need'
        self.cid_button_press = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.cid_key_press = self.fig.canvas.mpl_connect('key_press_event', self.on_keypressed)
        self.cid_pick = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        print('connecting to canvas')
        plt.show() 
        
    # Simple mouse click function to store coordinates
    def on_button_press(self,event):

        '''
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        '''

        if (event.button == 1):
            # Left button pressed - insert a new point

            'make sure the mouse is within the figure'
            if event.inaxes == None:
                print('pressed out of bounds!')
                return

            'insert the new coordinates tuple to repo'
            x_coord = int(event.xdata + .5)
            y_coord = int(event.ydata + .5)
            self.point_ID = self.point_ID + 1
            print('adding point x={}, y={}'.format(x_coord, y_coord))
            border_point = plt.plot([x_coord], [y_coord], marker='o', markersize=3, color="red", picker=5) #5 points tolerance
            #border_point = plt.plot([x_coord], [y_coord], marker='o', markersize=3, color="red")

            self.coords.append(tuple((x_coord, y_coord, self.point_ID, border_point)))

            'sort the coordinates and re-draw the border'
            self.coords = sorted(self.coords, key=getKey)

            're-draw the border line'
            new_border_line = [(i[0], i[1]) for i in self.coords]  # get first 2 elemens in the tuple
            self.border_line.set_data(*zip(*new_border_line))  # update the border line with the new coordinate
            plt.draw()

    
    def erase_last_point(self):
        'remove the last point'
        last_point = self.coords[-1:] # get the last point in the border list
        print('remove last point')
        last_point_handle = [i[3] for i in last_point] # extract the point handle (list) from the tuple        
        line2D = last_point_handle.pop(0) #pop the 1st line2D handle from the list
        line2D[0].remove() # remove the point (will be shown in the next draw)        
                    
        'delete last point from list'
        del self.coords[-1]
                
        're-draw the line'
        new_border_line = [(i[0],i[1]) for i in self.coords] # get first 2 elemens in the tuple
        self.border_line.set_data(*zip(*new_border_line)) #update the border line with the new coordinate
        plt.draw()

    def on_pick(self, event):

        if event.mouseevent.button == 3:
            # Right button picking
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            points = tuple(zip(xdata[ind], ydata[ind]))
            print('erase point: ', points)
            #print('remove:', event.artist)

            for i, point in enumerate(self.coords):
                if point[0] == xdata:
                    point_index = i

            #print('erase point number {}'.format(point_index))
            artist = event.artist
            artist.remove()

            'delete the point from list'
            del self.coords[point_index]

            're-draw the line'
            new_border_line = [(i[0], i[1]) for i in self.coords]  # get first 2 elemens in the tuple
            self.border_line.set_data(*zip(*new_border_line))  # update the border line with the new coordinate
            plt.draw()

            #point_for_delete = [item for item in self.coords if item[0] == xdata]

    def on_keypressed(self,event):

        if event.key == 'enter':
            'save coordinates to file'
            if os.path.exists(self.out_filename):
                print('removing the original CSV file')
                os.remove(self.out_filename)
            with open(self.out_filename,'w', newline='', encoding="utf-8") as f_output:
                csv_output = csv.writer(f_output)
                csv_output.writerows(self.coords)
                print('writing csv file to' + self.out_filename)
                
                'move the image file to annotated directory'
                if not(os.path.exists(self.image_file)):
                    shutil.move(self.image_file, self.output_dir)
                    print('image file moved to annotated directory')

            'disconnect from all events'
            print('disconnecting from canvas')
            self.fig.canvas.mpl_disconnect(self.cid_button_press) 
            self.fig.canvas.mpl_disconnect(self.cid_key_press)
            self.fig.canvas.mpl_disconnect(self.cid_pick)
            plt.close(1)                  
        
        elif event.key == 'z':            
            self.erase_last_point()


def annotate_image(image_filename):
    img = mpimg.imread(image_filename)
    fig, ax = plt.subplots()

    'maximize the window used'
    print('matplotlib backend: ' + get_backend())
    figManager = plt.get_current_fig_manager()
    figManager.window.state('withdrawn')  # maximize for TkAgg backend
    # figManager.window.showMaximized() # maximize for spider backend

    ax.imshow(img)
    border_extractor = ExtractCoords(fig, image_filename)
    border_extractor.connect()

def main(image_filename):

    # Check if an annotation file already exists
    if os.path.isfile(image_filename):
        annotate_image(image_filename)
    else:
        print('not an image file')


if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    # image_filename = askopenfilename(initialdir='/media/vision/In Process/ForAnnotation/GC23_fixes')  # show an "Open" dialog box and return the path to the selected file
    image_filename = askopenfilename(
        initialdir='/media/vision/In Process/ForAnnotation/(100,16)ImportanceCheck',
        filetypes=[('JPG file', '*.jpg')])  # show an "Open" dialog box and return the path to the selected file
    root.destroy()
    print('annotate image - ' + image_filename)

    main(image_filename)