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
* z key = remove last point
* enter key  = save to csv file, move the original image to "annotated" directory and quit
*
* @author: Ran Zaslavsky
*
'''

import matplotlib.pyplot as plt
import csv
import shutil
import os

' define key for coordinates sort '
def getKey(item):
    return(item[0])

class ExtractCoords:
    def __init__(self,my_figure,image_filename):
        
        'preparing output dir & filename'
        image_name = os.path.basename(image_filename)
        annotation_name = image_name.replace('.png','.csv')
        annotation_name = annotation_name.replace('.jpg','.csv')             
                
        'if annotated directory does not exist create it'        
        output_dir = os.path.dirname(image_filename) + '/annotated'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('outpur dir created')
        annotations_filename = os.path.join(output_dir, annotation_name)
        print('output file - ', annotations_filename)
        
        'initializing internal data'        
        self.coords = [] #list of coordinates tuple
        self.fig = my_figure
        self.axes = my_figure.axes
        self.image_file = image_filename
        self.output_dir = output_dir
        self.out_filename = annotations_filename
        self.point_ID = 0
        print('init coordinates extractor')
        
        'prepare the border line'
        self.border_line, = plt.plot(*zip(*self.coords[0:1]), 'r-')
   
        #axes = plt.gca()
        #line, = axes.plot(xdata, ydata, 'r-')
            
    def connect(self):
        'connect to all the events we need'
        self.cid_button_press = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.cid_key_press = self.fig.canvas.mpl_connect('key_press_event', self.on_keypressed)
        print('connecting to canvas')
        plt.show() 
        
    # Simple mouse click function to store coordinates
    def on_button_press(self,event):
        
        'if right key is pressed and close enough to an existing point, drag and drop it'
        '''
        if event.key not in ('right', 'left'):
            return
        '''
        
        'make sure the mouse is within the figure'
        #if event.inaxes != self.axes: return
        
        'insert the new coordinates tuple to repo'
        x_coord = int(event.xdata+.5)
        y_coord = int(event.ydata+.5)
        self.point_ID = self.point_ID + 1
        print('x={}, y={}'.format(x_coord, y_coord))
        border_point = plt.plot([x_coord], [y_coord], marker='o', markersize=3, color="red")
        #print(border_point)

        self.coords.append(tuple((x_coord,y_coord,self.point_ID,border_point)))
        
        'sort the coordinates and re-draw the border'
        self.coords = sorted(self.coords, key=getKey)
        
        're-draw the border line'
        new_border_line = [(i[0],i[1]) for i in self.coords] # get first 2 elemens in the tuple
        self.border_line.set_data(*zip(*new_border_line)) #update the border line with the new coordinate
        plt.draw()

    
    def erase_last_point(self):
        'remove the last point'
        last_point = self.coords[-1:] # get the last point in the border list
        print('remove last point')
        last_point_handle = [i[3] for i in last_point] # extract the point handle (list) from the tuple        
        line2D = last_point_handle.pop(0) #pop the 1st line2D handle from the list
        line2D[0].remove() # remove the point (will be shown in the next draw)        
                    
        'delete last point from list'
        #print(self.coords)
        del self.coords[-1]
        #print(self.coords)     
                
        're-draw the line'
        new_border_line = [(i[0],i[1]) for i in self.coords] # get first 2 elemens in the tuple
        self.border_line.set_data(*zip(*new_border_line)) #update the border line with the new coordinate
        plt.draw()

    
    def on_keypressed(self,event):

        if event.key == 'enter':
            'save coordinates to file'             
            with open(self.out_filename,'w', newline='', encoding="utf-8") as f_output:
                csv_output = csv.writer(f_output)
                csv_output.writerows(self.coords)
                print('writing csv file to' + self.out_filename)
                
                'move the image file to annotated directory'
                shutil.move(self.image_file, self.output_dir)
                print('image file moved to annotated directory') 

            'disconnect from all events'
            print('disconnecting from canvas')
            self.fig.canvas.mpl_disconnect(self.cid_button_press) 
            self.fig.canvas.mpl_disconnect(self.cid_key_press) 
            plt.close(1)                  
        
        elif event.key == 'z':            
            self.erase_last_point()
                                    
