# Package constants
import numpy as np

# Caffe related
MEMORY_DATA_LAYER = 29  # matches definition in caffe.proto
STATE_MD_LAYER = 0
NEXT_STATE_MD_LAYER = 1
ACTION_REWARD_MD_LAYER = 2

NUM_ACTIONS = 4
DTYPE = np.float32
DTYPE_SIZE = 4

MSG_LENGTH = 1
GRAD_UPDATE = 'G'
DARWIN_UPDATE = 'D'
