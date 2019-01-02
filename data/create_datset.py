
'''
*
* create_dataset()
*
* Purpose: copy train/valid/test/meta_data from a specified directory (built in a required
*          structure to an output "Dataset" directory.
*
* Inputs:
*   Repositor & output Dataset folders
*
* Outputs:
*   train/valid/test/meta_data folders
*   aggregated meta_data CSV file
*
* Written by: Ran Zaslavsky 30-12-2018
'''



import glob, os
import shutil
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog


def main(data_dir, target_dir):

    # Gather file paths to all annotated images within the target directory (and subfolders)
    object_dirs = glob.glob(data_dir + '/*')

    objects = {}
    for d in object_dirs:
        objects[d.split('/')[-1]] = glob.glob(d + '/*.csv') # Take the last subfolder name as the key
        #objects[d.split('/')[1]] = glob.glob(d + '/*.csv')
        print(d)

    # Create train/valid/test directories to store our TFRecords
    if not os.path.exists(target_dir) and not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    if not os.path.exists(target_dir+'/train') and not os.path.isdir(target_dir+'/train'):
        os.mkdir(target_dir+'/train')

    if not os.path.exists(target_dir+'/valid') and not os.path.isdir(target_dir+'/valid'):
        os.mkdir(target_dir+'/valid')

    if not os.path.exists(target_dir + '/test') and not os.path.isdir(target_dir + '/test'):
        os.mkdir(target_dir + '/test')

    if not os.path.exists(target_dir + '/meta_data') and not os.path.isdir(target_dir + '/meta_data'):
        os.mkdir(target_dir + '/meta_data')

    object_names = list(objects.keys())
    copy_dir_list = ['/train', '/valid', '/test']
    master_filename = os.path.join(target_dir + '/meta_data', "meta_data.csv") # defines the master CSV file name

    # Copy TFRecord files
    for object in object_names:
        object_path = os.path.join(data_dir, object)
        for dir_name in copy_dir_list:
            tfrec_files = glob.glob(object_path + dir_name + '/*.tfrecord')
            for file in tfrec_files:
                shutil.copy(file, target_dir + dir_name)

    # Copy CSV files & create a master CSV file    
    fout = open(master_filename, "a")
    #headers = []
    #headers.append(['frame', 'label', 'for_use', 'frame_type'])
    #headers = 'frame,label,for_use,frame_type'
    #fout.write(headers)
    for object in object_names:
        object_path = os.path.join(data_dir, object)
        data_files = glob.glob(object_path + '/meta_data' + '/*.csv')
        for file_ in data_files:
            shutil.copy(file_, target_dir + '/meta_data')
            # Copy CSV lines to master CSV
            for line in open(file_):
                fout.write(line)
    fout.close()

    # Analyze the master CSV
    master_df = pd.read_csv(master_filename, names=["frame","label","for_use","frame_type"])
    table = pd.pivot_table(master_df, index=['label'], values=['frame'], columns=['frame_type'], aggfunc = [len], margins=True)
    #table = pd.pivot_table(master_df, index=['label'], values=['frame'], aggfunc=[len], margins=True)
    writer = pd.ExcelWriter(os.path.join(target_dir + '/meta_data',"meta_data_pivot.xlsx"))
    table.to_excel(writer)
    writer.save()


    #meta_data.extend() # append the new frame labels data

if __name__ == '__main__':

    ' when executed as a script, open a GUI window to select the presented TFrecord file '
    '''
    root = tk.Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    data_dir = filedialog.askdirectory()
    '''

    data_dir = '/media/vision/DataRepo'
    #data_dir = '/media/vision/temp_DataRepo'
    target_dir = '/media/vision/Datasets/Dataset_5'

    #data_dir = 'annotated' # TEMPORARY !!!!!!
    print('Extract files from DataRepo - ' + data_dir + ':')

    main(data_dir, target_dir)