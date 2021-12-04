import config
import os

# General:
# Adadelta optimizer configuration
learning_rate=0.005
# leearning_rate = 1.0
rho=0.9
epsilon=1e-07

# BNN weight regularisation:
tau = 0.01
dropout = 0.2

# Settings for mosquito event detection:
epochs = 2
batch_size = 32
lengthscale = 0.01

# Settings for multi-species as in paper:
batch_size = 128 
epochs = 10



# Make output sub-directory for saving model
directory = os.path.join(config.model_dir, 'keras')
if not os.path.isdir(directory):
	os.mkdir(directory)
	print('Created directory:', directory)



