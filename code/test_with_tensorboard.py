import tensorflow as tf
import os
import numpy as np
from TF_stixels.code.model import model_fn, params
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python import debug as tf_debug # for debugging

import tkinter as tk
from tkinter.filedialog import askopenfilename

# params
H = 370
W = 24
C = 3

image_filename = './stixel_010_L11.png'

'''
file_input = tf.placeholder(tf.string, ())
print('image file = ',file_input)
image = tf.image.decode_png(tf.read_file(file_input))
images = tf.expand_dims(image,0)
#images = tf.cast(images, tf.float32)
images = tf.cast(images, tf.float32) / 128. - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(image, (W, H))
inputs = images
'''


# another option:
'''
with open(image_filename, 'rb') as f:
    img_data = f.read()
    '''

# Format the data
'''
images = tf.image.decode_png(img_data, channels=3, dtype=tf.uint8) # Decode the raw bytes so it becomes a tensor with type.
images = tf.cast(images, tf.float32)  # The type is now uint8 but we need it to be float.
inputs = np.reshape(images, [-1, H, W, C])
'''

'''
img_data = mpimg.imread(image_filename)
image = tf.image.decode_png(img_data, channels=C)
image = tf.expand_dims(image, 0)
#image = tf.image.resize_bilinear(image, [H, W], align_corners=False)
image = tf.squeeze(image, squeeze_dims=[0])
image = tf.cast(image, dtype=tf.uint8)
inputs = np.reshape(image, [-1, H, W, C])
'''


'''
def decode_and_resize(image_str_tensor):
    """Decodes jpeg string, resizes it and returns a uint8 tensor."""
    image = tf.image.decode_png(image_str_tensor, channels=C)
    image = tf.expand_dims(image, 0)
    #image = tf.image.resize_bilinear(image, [H, W], align_corners=False)
    image = tf.squeeze(image, squeeze_dims=[0])
    image = tf.cast(image, dtype=tf.uint8)
    return image
    '''


print(os.getcwd())


# Code that works but prediction is constnat!!!
root = tk.Tk()
root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
image_filename = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
print('image file - ' + image_filename)

images = mpimg.imread(image_filename)
fig, ax = plt.subplots()
ax.imshow(images)


#images = images/255.0
images = np.float32(images)
inputs = np.reshape(images, [-1, H, W, C])

#print(inputs.shape())

# Determine the model to be used for inference
model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2018-12-16_17-39-06' # 25 epochs + using leakyrelu
#model_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2018-12-16_14-58-28'

# Load the model
estimator = tf.estimator.Estimator(model_fn, model_dir=model_dir, params=params)

# Create the input_fn
input_fn = tf.estimator.inputs.numpy_input_fn(x={'image' : inputs}, num_epochs=1, shuffle=False)

# Prepare hooks for debugging
hooks = [tf_debug.TensorBoardDebugHook(grpc_debug_server_addresses="dev:6064")]
#hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline")] # Hooks to the manual debugger
#hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline" "dev:6064")]

# Predict!
predictions = estimator.predict(input_fn=input_fn, hooks=hooks)
#predictions = estimator.predict(input_fn=input_fn)
predictions_list = []
predictions_list = list(predictions)
predicted_label = predictions_list[0]
print('prediction = {}'.format(predicted_label))
#print('max = {}'.format(predictions_list[np.argmax(predictions_list)]))

# Print tensorboard data
print('tensorboard --logdir=' + str(model_dir) + '--port 6006 --debugger_port 6064')

'''
plt.plot(12, predicted_label * 5, marker='o', markersize=4, color="red")
plt.draw()
#plt.show()
'''


'''
with tf.Session() as sess:
    classification = sess.run(estimator.predict(input_fn=input_fn))
'''


# this is how to get your results:
#predictions_dict = next(predictor)
#print(list(predictions_dict))

'''
for single_prediction in predictions:
    predicted_class = single_prediction['class']
    probability = single_prediction['probability']
    #do_something_with(predicted_class, probability)
'''
