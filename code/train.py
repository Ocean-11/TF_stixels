'''
*
* train
*
* Purpose: the module trains a MobileNetV2 stixels model
*
* Inputs:
*
*
* Outputs:
*
*
* Conventions: (x=0, y=0) is the upper left corner of the image
*
* Written by: Ran Zaslavsky 10-12-2018 (framework originates from excellent https://crosleythomas.github.io/blog/)
*
'''

import tensorflow as tf
import glob
import os
import shutil
from tensorflow.python import debug as tf_debug # RAN


#############################
###   Gather Saved Data   ###
#############################

train_dir = '../data/annotated/tfrecords/train'
valid_dir = '../data/annotated/tfrecords/valid'
test_dir = '../data/annotated/tfrecord/test'

'''
train_dir = '../data/tfrecords/train'
valid_dir = '../data/tfrecords/valid'
test_dir = '../data/tfrecords/test'
'''

train_files = glob.glob(train_dir + '/*.tfrecord')
valid_files = glob.glob(valid_dir + '/*.tfrecord')
test_files = glob.glob(test_dir + '/*.tfrecord')

# Now make all these files accessible depending on whether we
#    are in training ('train') or validation ('valid') mode.
data_files = {'train' : train_files, 'valid' : valid_files, 'test' : test_files}

#plt.style.use("seaborn-colorblind")
# RAN - Setup logger - only displays the most important warnings
tf.logging.set_verbosity(tf.logging.INFO) # possible values - DEBUG / INFO / WARN / ERROR / FATAL

#######################################
###   Creating a dataset_input_fn   ###
#######################################

'''
    dataset_input_fn - Constructs a Dataset, Iterator, and returns handles that will be called when
    Estimator requires a new batch of data. This function will be passed into 
    Estimator as the input_fn argument.

    Inputs:
        mode: string specifying whether to take the inputs from training or validation data
    Outputs:
        features: the columns of feature input returned from a dataset iterator
        labels: the columns of labels for training return from a dataset iterator
'''
def dataset_input_fn(mode):
    # Function that does the heavy lifting for constructing a Dataset
    #    depending on the current mode of execution
    dataset = load_dataset(mode)
    # Making an iterator that runs from start to finish once
    #    (the preferred type for Estimators)
    iterator = dataset.make_one_shot_iterator()
    # Consuming the next batch of data from our iterator
    features, labels = iterator.get_next()
    return features, labels

####################################
###   Constructing the Dataset   ###
####################################

'''
    load_dataset() - Loads and does all processing for a portion of the dataset specified by 'mode'.

    Inputs:
        mode: string specifying whether to take the inputs from training or validation data

    Outputs:
        dataset: the Dataset object constructed for this mode and pre-processed
'''
def load_dataset(mode):
    # Taking either the train or validation files from the dictionary we constructed above
    files = data_files[mode]
    # Created a Dataset from our list of TFRecord files
    dataset = tf.data.TFRecordDataset(files)
    # Apply any processing we need on each example in our dataset.  We
    #    will define parse next.  num_parallel_calls decides how many records
    #    to apply the parse function to at a time (change this based on your
    #    machine).
    dataset = dataset.map(parse, num_parallel_calls=2)
    # Shuffle the data if training, for validation it is not necessary
    # buffer_size determines how large the buffer of records we will shuffle
    #    is, (larger is better!) but be wary of your memory capacity.
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=1000)
    # Batch the data - you can pick a batch size, maybe 32, and later
    #    we will include this in a dictionary of other hyper parameters.
    dataset = dataset.batch(params.batch_size)
    return dataset


###################################
###   Define Output Directory   ###
###################################

import time, datetime
from model import model_fn, params, config, parse

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

model_dir = '../results/' + timestamp + '_LR_' + str(params.learning_rate)+ '_EP_' + str(params.train_epochs)
#print(model_dir)

#############################
###    Define Estimator   ###
#############################

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config, params=params)

##########################
###   Copy File Over   ###
##########################

# Find the path of the current running file (train script)
curr_path = os.path.realpath(__file__)
model_path = curr_path.replace('train.py', 'model.py')

# Define the path of your factored out model.py file
#model_file = '/some/path/model.py'
model_file = './model.py'

# Now copy the training script and the model file to
#   model_dir -- the same directory specified when creating the Estimator

# Note: copy over more files if there are other important dependencies.
os.mkdir(model_dir)
shutil.copy(curr_path, model_dir)
shutil.copy(model_path, model_dir)

# Create a LocalCLIDebugHooks and use it as a monitor when calling fit()
hooks = [tf_debug.TensorBoardDebugHook(grpc_debug_server_addresses="dev:6064")]
#hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline")] # Hooks to the manual debugger

# Training/Evaluation Loop
for e in range(params.train_epochs):
    print('Epoch: ' + str(e))
    #estimator.train(input_fn=lambda: dataset_input_fn('train'), hooks=hooks) # RAN
    estimator.train(input_fn=lambda: dataset_input_fn('train'))
    print('### validate ###')
    estimator.evaluate(input_fn=lambda: dataset_input_fn('valid'))

print('tensorboard --logdir=' + str(model_dir))
