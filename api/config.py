RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data'
CKPT_PATH = 'api/model/checkpoints'

# Training
BATCH_SIZE = 64
EPOCHS = 50

# Transformer
NUM_LAYERS = 4
EMBEDDING_DIMS = 128
NUM_HEADS = 8
EXPANDED_DIMS = 512

# Preprocessing
PRELOAD_DATA = False
FILTER_WORDS = ['removed', 'deleted', 'tl;dr']

# ALT GENERATOR
MODEL_NAME = "hot-take-model"
SENTENCEPIECE_MODEL = "hot-take-model/1/assets/wmtende.model"
TIMEOUT = 5
MAX_LENGTH = 100