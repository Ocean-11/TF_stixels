# TF_stixels
This folder contains code for Stixels annotation, training  and prediction. The framework
 originates from Shaharzuler/NexarStixelnet excellent implementation of StixelNet for mobile

# Folders structure
data - includes all the relevant scripts for images annotation, visualization and TFRecords
 creation.

code - includes model, training & prediction scripts


## data folder
crop_image()
* crop image allows cropping the an entire folder of images, creating a "cropped" images folder 

extract_border_coords()
* The script allows selection of an image to be annotated. In case an annotation file already
 exists, the annotations would be presented on top of the image. Annotations may be created &
  edited via a dedicated GUI. Once done the annotation CSV file is created/saved

correct_annotation()
* the script allows selection of a control image (with wrong annotations), finds the original
 image + CSV file and  uses extract_border_coords to correct it

folder2TFRec()
* The module receives a directory name, creates train/valid/test/control/meta_data folders,
 and scans it's "annotated" folder, translating annotated images into train/valid/test TFrecords,
  based on thye required stixels width (W). Control images and meta_data are also saved
* Note that folder2TFRec2() is used to create prediction-ready TFRecords and encapsulates
frame_id's rather than lables. 
* Future: include both frame_id, label, image_name and other features
 in the same TFrecord file 

tree2TFRec()
* The module uses folder2TFRec() to create TFrecords for an entire folders tree   

create_new_dataset()
* copy train/valid/test/meta_data from a specified directory (built in a required structure
 to an output "Dataset" directory.

## code folder
model()
* the module implements a MobileNetV2 stixels model

model_for_CRF()
* the same MobileNetV2 stixels model implementation outputting the Softmax probabilities as well 

train()
* the module trains a MobileNetV2 stixels model

predict()/predict_folder()
* the module selects a trained model directory, scans a test images directory and annotates
 the images with both labels & predictions

CRF())
* the module implements a Conditional Random Fields filter

video()
* the module creates a video from an images folder                   

## General Files
image_predict()
* the module analyzes an image with and w/o GT file and produces, implements the inference,
 and produces an output image + meta data 