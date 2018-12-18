import tensorflow as tf
import os
import numpy as np
from TF_stixels.code.model import model_fn, params
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python import debug as tf_debug # for debugging
from PIL import Image
import cv2
import re

import tkinter as tk
from tkinter.filedialog import askopenfilename

# Params
H = 370
W = 24
C = 3

#image_filename = '../data/Lawn_stixels/Garden_8/stixel_082_L02.png'
#image_filename = '../data/Lawn_stixels/Garden_8/stixel_087_L38.png'
#image_filename = '../data/Lawn_stixels/Garden_8/stixel_087_L03.png'
#image_filename = '../data/Lawn_stixels/Garden_8/stixel_084_L02.png'
#image_filename = '../data/Lawn_stixels/Garden_8/stixel_000_L35.png'
#image_filename = './stixel_010_L11.png'
#image_filename = './stixel_088_L38.png'


# Code that works but prediction is constnat!!!
root = tk.Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
image_filename = askopenfilename(initialdir='../data/Lawn_stixels/Garden_8/')  # show an "Open" dialog box and return the path to the selected file
print('image file - ' + image_filename)

# Read the image file
image = cv2.imread(image_filename)

' display the current frame'
fig, ax = plt.subplots()
ax.imshow(image)

# Resize the image to the desired size and pre-process exactly as in the training
images = []
image = cv2.resize(image, (W,H))
images.append(image)
images = np.array(images, dtype=np.uint8)
images = np.float32(images)
#images = images.astype('float32')
#images = np.multiply(images, 1.0/128.0-1)

# The imput of the network is of shape [None,HmW,C] so we reshape
inputs = np.reshape(images, [-1, H, W, C])

# Determine the model to be used for inference
model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2018-12-17_18-33-01' # 1000 epochs + using leakyrelu
#model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2018-12-18_13-07-00' # 100 epochs + using relu6 + data normailization
#model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2018-12-16_17-39-06' # 25 epochs + using leakyrelu
#model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2018-12-16_14-58-28'

# Load the model
estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

# Create the input_fn
input_fn = tf.estimator.inputs.numpy_input_fn(x={'image' : inputs}, num_epochs=1, shuffle=False)

# Prepare hooks for debugging
hooks = [tf_debug.TensorBoardDebugHook(grpc_debug_server_addresses="dev:6064")]

# Predict!
#predictions = estimator.predict(input_fn=input_fn, hooks=hooks)
predictions = estimator.predict(input_fn=input_fn)
predictions_list = []
predictions_list = list(predictions)
predicted_label = predictions_list[0]
print('prediction = {}'.format(predicted_label))
#print('max = {}'.format(predictions_list[np.argmax(predictions_list)]))

# Print tensorboard data
print('tensorboard --logdir=' + str(model_dir) + '--port 6006 --debugger_port 6064')

# Save stixel with prediction
plt.plot(12, predicted_label * 5 + 2, marker='o', markersize=5, color="red")
stixel_name = str(os.path.basename(image_filename))
save_fig_name = stixel_name.replace('.png','_P'+ str(predicted_label) + '.png')
plt.savefig('../data/Lawn_stixels/'+save_fig_name)

