# TF_stixels
This library contains code for Stixels annotation, training  and prediction. The framework
 originates from Shaharzuler/NexarStixelnet excellent implementation of StixelNet for mobile devices

## Folders structure
* code - includes the stixel model, training & prediction scripts

* data - images/data preparation scripts & training datasets repository

* embedded - optimized scripts for embedded prediction

* results - training outcomes (including used train/model) repository

## Root Directory Scripts
default_GT.csv()
* default ground truth to ve used when displaying images that were not annotated

folder_predict.py()
* Scans an entire folder and analyzes the images within it using image_predict() script

image_predict.py()
* the module analyzes an image using a defined model for inference, producing an output image + meta data

## Code folder
crf.py()
* implements a Conditional Random Fields filter operated on the stixels outcome

folder2TFRec.py()
* receives a directory name, scans it's "annotated" folder, and translates annotated images into
 stixels train/valid/test TFrecords. Control images and meta data are also created

model.py()
* the module implements a MobileNetV2 stixels model, outputing ArgMax only..

model_for_CRF.py()
* the module implements a MobileNetV2 stixels model, outputing ArgMax and Softmax probabilities

model_for_CRF_quant.py()
* the module implements a MobileNetV2 stixels model, outputing ArgMax and Softmax probabilities (still under debugging)

train.py()
* trains a MobileNetV2 stixels model

predict()/predict_folder()
* the module selects a trained model directory, scans a test images directory and annotates
 the images with both labels & predictions

video()
* the module takes an images folder and turns it into a video

## data folder
color_to_bw()
* translates an images folder to grey scale

correct_annotation()
* the script allows selection of a control image (with wrong annotations), finds the original
 image + CSV file and  uses extract_border_coords to correct it

create_new_dataset()
* copy train/valid/test/meta_data from a specified directory (built in a required structure
 to an output "Dataset" directory.
 
crop_image()
* crop image supports cropping the an entire folder of images, creating a "cropped" images folder 

extract_border_coords()
* The script allows selection of an image to be annotated. In case an annotation file already
 exists, the annotations would be presented on top of the image. Annotations may be created &
  edited via a dedicated GUI. Once done the annotation CSV file is created/saved

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

## Embedded folder

Includes an internal trained stixelNet model to be used for prediction

gc_crf()
* Random Fields filter adapted for an embedded grass classifer

gc_eff_crf()
* Random Fields filter optimized to run from an embedded grass classifer utilizing only the ArgMax

gc_image2TFRec()
* translates an image into a TFrecord ready to be used for prediction

gc_predictor()
* estimates the border using a stixelnet predictor and CRF filter

