
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


def main(data_dir, x_min, x_max):

    # Gather annotated images paths
    object_name = os.path.basename(os.path.normpath(data_dir))
    objects = glob.glob(data_dir + '/annotated/*.jpg')
    num_images = len(objects)

    # In case no there are no annotations, return
    if num_images == 0:
        print('no images within ' + object_name)
        return
    else:
        print('Folder name: ' + object_name + ' ,images to be converted - {}'.format(num_images))

    # Create required folders
    data_dir = data_dir+'_BW'
    print(data_dir)
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    data_dir = data_dir+'/annotated'
    if os.path.exists(data_dir) and os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)


    # Go through object images
    for index in range(num_images):

        frame_path = objects[index]
        '''
        rint('crop  {}'.format(frame_path))
        img = mpimg.imread(frame_path)
        height, width, c = img.shape
        cropped_img = img[0:height + height, x_min:x_max, :]
        image_file_name = os.path.join(data_dir, os.path.basename(frame_path))
        from PIL import Image
        im = Image.fromarray(cropped_img)
        im.save(image_file_name)
        '''

        from PIL import Image
        print('translate  {} to B&W'.format(frame_path))
        image_file = Image.open(frame_path)  # open colour image
        image_file_name = os.path.join(data_dir, os.path.basename(frame_path.replace('.jpg','_BW.jpg')))
        image_file = image_file.convert('L')  # convert image to black and white
        image_file.save(image_file_name)
        if os.path.exists(frame_path.replace('.jpg','.csv')):
            shutil.copy(frame_path.replace('.jpg','.csv'),os.path.join(data_dir,os.path.basename(frame_path).replace('.jpg','_BW.csv')))



if __name__ == '__main__':

    # Params
    x_min = 130
    x_max = 530

    ' when executed as a script, open a GUI window to select the presented TFrecord file '
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    data_dir = filedialog.askdirectory(initialdir='/media/vision/DataRepo/')
    root.destroy()

    print('Translate images within - ' + data_dir + ':')
    #print('Crop images within - ' + data_dir + ':')

    main(data_dir, x_min, x_max) # True saves a control image to a control directory
