NUM_EPOCHS = 50
MODE = 4
EPSILON = 0.15
MAX_LK_ITERATIONS = 60
VOT_ROOT_DIR = '../../data/VOT/'
ALOV_ROOT_DIR = '../../data/ALOV/'
VALIDATION_SPLIT = 0.3
SHUFFLE_TRAIN = True
RANDOM_SEED = 42
BATCH_SIZE = 1
LR = 0.0005
MOMENTUM = 0.9
L2 = 0.0001
TRAIN_EXAMPLES = 20000
CUDA=True
CONTEXT_AMOUNT = 0
EXEMPLAR_SIZE = 127
INSTANCE_SIZE = 255
TRANSITION_LR = 0.95


# DEVICE = if torch.cuda.is_available() torch.device("cuda:0") else torch.device("cpu")