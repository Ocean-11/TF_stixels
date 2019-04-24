
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

def main(data_dir, target_dir, stixel_width, stixel_height):

    # Gather file paths to all annotated images within the target directory (and subfolders)
    object_dirs = glob.glob(data_dir + '/*')

    objects = {}
    for d in object_dirs:
        objects[d.split('/')[-1]] = glob.glob(d + '/*.csv') # Take the last subfolder name as the key
        #objects[d.split('/')[1]] = glob.glob(d + '/*.csv')
        print('copy from: ' + d)

    # Create train/valid/test directories to store our TFRecords
    print('target directory = ' + target_dir)
    if not os.path.exists(target_dir) and not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    # Create required folders
    folders = {'train', 'valid', 'test', 'meta_data'}
    for folder_ in folders:
        new_folder = target_dir + '/' + folder_
        if not os.path.exists(new_folder) and not os.path.isdir(new_folder):
            os.mkdir(new_folder)

    object_names = list(objects.keys())
    dim_alias = '/H' + str(stixel_height) + '_W' + str(stixel_width)
    copy_dir_list = ['/train', '/valid', '/test']
    master_filename = os.path.join(target_dir + '/meta_data', "meta_data.csv") # defines the master CSV file name

    # Copy TFRecord files
    for object in object_names:
        object_path = os.path.join(data_dir, object)
        for dir_name in copy_dir_list:
            tfrec_files = glob.glob(object_path + dim_alias + dir_name + '/*.tfrecord')
            for file in tfrec_files:
                print('copy ' + file + ' to ' + dir_name)
                shutil.copy(file, target_dir + dir_name)


    # Copy CSV files & create a master CSV file
    fout = open(master_filename, "a")
    for object in object_names:
        object_path = os.path.join(data_dir, object)
        data_files = glob.glob(object_path + dim_alias + '/meta_data' + '/*.csv')
        for file_ in data_files:
            print('copy ' + file_)
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

    #data_dir = '/media/vision/DataRepo'
    data_dir = '/media/dnn/ML/DataRepo'
    #target_dir = '/media/vision/Datasets/Dataset_19'
    target_dir = '/media/dnn/ML/Datasets/Dataset_24_222_1'
    stixel_width = 24 # 36
    stixel_height = 222 # 370
    print('Extract files from DataRepo - ' + data_dir + ':')

    main(data_dir, target_dir, stixel_width, stixel_height )