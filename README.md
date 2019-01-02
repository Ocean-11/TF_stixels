# TF_stixels
This folder contains code for Stixels annotation, training  and prediction. The framework originates from Shaharzuler/NexarStixelnet excellent implementation of StixelNet for mobile

# Folders structure
data - includes all the relevant scripts for images annotation, visualization and TFRecords creation.

code - includes model, training & prediction scripts


## data folder
annotate_new_image()
*   The script selects an image, provides GUI for annotation, and once done creates an annotation CSV file and copies both to the "annotated" folder. annotate_new_image() uses extract_border_coords() as a Helper function

show_image_annotations()
*   The script selects an image, opens the accompanying CSV annotations file and displays the annotations on top of the image

folder2tTFRec()
*   The module receives a directory name, creates train/valid/test/control/meta_data olders, and scans it's "annotated" folder, translating annotated images into train/valid/test TFrecords. Control images and meta_data are also saved

## code folder

create_dataset()
*   copy train/valid/test/meta_data from a specified directory (built in a required structure to an output "Dataset" directory.

train()
*   the module trains a MobileNetV2 stixels model

model()
*   the module implements a MobileNetV2 stixels model

predict_folder()
* the module selects a trained model directory, scans a test images directory and annotates the images with both labels & predictions
                   
