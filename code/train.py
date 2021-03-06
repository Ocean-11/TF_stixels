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
import sys

######################################################
### Define InitHook to support single warm startup ###
######################################################

# based on - https://stackoverflow.com/questions/49846207/tensorflow-estimator-warm-start-from-and-model-dir

class InitHook(tf.train.SessionRunHook):
    """initializes model from a checkpoint_path
    args:
        modelPath: full path to checkpoint
    """
    def __init__(self, checkpoint_dir):
        self.modelPath = checkpoint_dir
        self.initialized = False

    def begin(self):
        """
        Restore encoder parameters if a pre-trained encoder model is available and we haven't trained previously
        """
        if not self.initialized:
            log = tf.logging.getLogger('tensorflow')
            checkpoint = tf.train.latest_checkpoint(self.modelPath)
            if checkpoint is None:
                log.info('No pre-trained model is available, training from scratch.')
            else:
                log.info('Pre-trained model {0} found in {1} - warmstarting.'.format(checkpoint, self.modelPath))
                tf.train.warm_start(checkpoint)
            self.initialized = True


#############################
###   Gather Saved Data   ###
#############################

dataset_dir = '/Dataset_24_222_2_screened'
train_dir = '../data' + dataset_dir + '/train'
valid_dir = '../data' + dataset_dir + '/valid'
test_dir = '../data' + dataset_dir + '/test'

'''
train_dir = '../data/Dataset_8/train'
valid_dir = '../data/Dataset_8/valid'
test_dir = '../data/Dataset_8/test'
'''

'''
train_dir = '../data/tfrecords/train'
valid_dir = '../data/tfrecords/valid'
test_dir = '../data/tfrecords/test'
'''

train_files = glob.glob(train_dir + '/*.tfrecord')
valid_files = glob.glob(valid_dir + '/*.tfrecord')
test_files = glob.glob(test_dir + '/*.tfrecord')

for file_ in valid_files:
    print(file_)

# Now make all these files accessible depending on whether we
#    are in training ('train') or validation ('valid') mode.
data_files = {'train' : train_files, 'valid' : valid_files, 'test' : test_files}

#plt.style.use("seaborn-colorblind")
# RAN - Setup logger - only displays the most important warnings
tf.logging.set_verbosity(tf.logging.INFO) # possible values - DEBUG / INFO / WARN / ERROR / FATAL
#tf.logging.set_verbosity(tf.logging.INFO)

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
        dataset = dataset.shuffle(buffer_size=4000) # RAN 07-01-2019
        #dataset = dataset.shuffle(buffer_size=1000)
    # Batch the data - you can pick a batch size, maybe 32, and later
    #    we will include this in a dictionary of other hyper parameters.
    dataset = dataset.batch(params.batch_size)
    return dataset


#######################################
###   Creating a dataset_input_fn   ###
#######################################

'''
    serving_input_receiver_fn - 

    Inputs:
        mode: string specifying whether to take the inputs from training or validation data
    Outputs:
        features: 
        receiver_tensors: 
'''

'''
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[1],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
  '''


###################################
###   Define Output Directory   ###
###################################y

import time, datetime
from model_for_CRF import model_fn, params, config, parse

ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

model_dir = '../results/' + timestamp + '_EP_' + str(params.train_epochs)
#print(model_dir)

# Make sure test file stixels width are compatible with the model stixels width
file_name = train_files[0]
file_name = file_name.split('/')[-1]
file_name = file_name.split(('.')[0])
file_name = file_name[0]
train_file_stixel_width = int(file_name.split('W')[-1])

if train_file_stixel_width != params.image_width:
    print('train images do not match the model stixels width!!! - {},{}'.format(train_file_stixel_width, params.image_width))
    sys.exit()

#############################
###    Define Estimator   ###
#############################

estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, config=config, params=params)

'''
# Code for warm start - does not work properly at initializes the variables on every epoch!!! 
warm_start_path = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-01_14-48-55_EP_100'
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   model_dir=model_dir,
                                   config=config,
                                   params=params, 
                                   warm_start_from=warm_start_path)
'''

'''
def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print [str(i.name) for i in not_initialized_vars] # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

'''

##########################
###   Copy File Over   ###
##########################

# Find the path of the current running file (train script)
curr_path = os.path.realpath(__file__)
model_path = curr_path.replace('train.py', 'model_for_CRF.py')

# Define the path of your factored out model.py file
#model_file = '/some/path/model_for_CRF.py'
model_file = './model_for_CRF.py'

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


def make_serving_input_receiver_fn():
    inputs = {'image': tf.placeholder(
        shape=[None, params.image_height, params.image_width, params.image_depth], dtype=tf.float32, name='serving_input_image')}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)


export_dir = os.path.join(model_dir, 'export')

if tf.gfile.Exists(export_dir):
    tf.gfile.DeleteRecursively(export_dir)

estimator.export_savedmodel(
    export_dir_base=export_dir,
    serving_input_receiver_fn=make_serving_input_receiver_fn(),
    strip_default_attrs=True)



# Export the trained estimator - XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn, strip_default_attrs=True)

#This method builds a new graph by first calling the serving_input_receiver_fn() to obtain feature
#Tensors, and then calling this Estimator's model_fn() to generate the model graph based on those
#features. It starts a fresh Session, and, by default, restores the most recent checkpoint into it.
#(A different checkpoint may be passed, if needed.) Finally it creates a time-stamped export
# directory below the given export_dir_base (i.e., export_dir_base/<timestamp>),
# and writes a SavedModel into it containing a single MetaGraphDef saved from this Session.





'''
' Code for warm start: '
warm_start_dir = '/home/dev/PycharmProjects/stixel/TF_stixels/results/2019-01-01_14-48-55_EP_100'
initHook = InitHook(checkpoint_dir = warm_start_dir)
trainSpec = tf.estimator.TrainSpec(
    input_fn = train_input_fn,
    max_steps = N_STEPS, 
    hooks = [initHook]
)
evalSpec = tf.estimator.EvalSpec(
    input_fn = eval_input_fn,
    steps = None,
    name = 'eval',
    throttle_secs = 3600
)
tf.estimator.train_and_evaluate(estimator, trainSpec, evalSpec)
'''

'''
experiment = tf.contrib.learn.Experiment(
    estimator=estimator,  # Estimator
    train_input_fn=train_input_fn,  # First-class function
    eval_input_fn=eval_input_fn,  # First-class function
    train_steps=params.train_steps,  # Minibatch steps
    min_eval_frequency=params.min_eval_frequency,  # Eval frequency
    train_monitors=[train_input_hook],  # Hooks for training
    eval_hooks=[eval_input_hook],  # Hooks for evaluation
    eval_steps=None  # Use evaluation feeder until its empty
)

learn_runner.run(
    experiment_fn=experiment_fn,  # First-class function
    run_config=run_config,  # RunConfig
    schedule="train_and_evaluate",  # What to run
    hparams=params  # HParams
)
'''
