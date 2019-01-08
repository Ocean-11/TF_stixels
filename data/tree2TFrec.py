

import tkinter as tk
from tkinter import filedialog
import glob
import folder2TFRec
from TF_stixels.code.model import params

root = tk.Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
data_dir = filedialog.askdirectory(initialdir='/media/vision/DataRepo')
root.destroy()

print('Convert to TFrecords all annotated images within - ' + data_dir + ':')

# Gather file paths to all annotated images within the target directory (and subfolders)
object_dirs = glob.glob(data_dir + '/*')

# Go through each subfolder and transfer it to TFrecords
objects = {}
for dir_ in object_dirs:
    #objects[d.split('/')[-1]] = glob.glob(d + '/*.csv')  # Take the last subfolder name as the key
    print(dir_)
    folder2TFRec.main(dir_, params.image_width, True)

