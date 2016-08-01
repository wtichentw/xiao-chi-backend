""" Dir to te images """
image_dir = '/Users/wtichen/Workspace/xiao-chi'

""" Dir to download original Inception v3 model """
raw_model_dir = '/tmp/imagenet'

""" Dir to put cached bottleneck value """
bottleneck_dir = '/tmp/bottleneck'

""" Dir to put training log for tensorboard """
summaries_dir = '/tmp/retrain_logs'

""" Name for retrained model and category label """
output_graph = '/tmp/output_graph.pb'
output_labels = '/tmp/output_labels.txt'

""" final softmax layer name for our category """
final_tensor_name = "final_result"

""" How many training steps to run before ending."""
how_many_training_steps = 40000

""" How large a learning rate to use when training."""
learning_rate = 0.001

""" How often to evaluate the training results."""
eval_step_interval = 10

""" How many images to train on at a time."""
train_batch_size = 100

""" How many images to test on at a time. This"""
""" test set is only used infrequently to verify"""
""" the overall accuracy of the model."""
test_batch_size = 500

""" How many images to use in an evaluation batch."""
validation_batch_size = 100

""" Percentage to split data for validate and test """
testing_percentage = 10 
validation_percentage = 10


DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
