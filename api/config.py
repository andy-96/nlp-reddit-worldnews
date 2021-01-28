RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data'
CKPT_PATH = 'api/model/checkpoints'

# Preprocessing
PRELOAD_DATA = False
FILTER_WORDS = ['removed', 'deleted', 'tl;dr']

# OpenNMT generator
MODEL_NAME = "hot-take-model"
SENTENCEPIECE_MODEL = "hot-take-model/1/assets/wmtende.model"
TIMEOUT = 5
MAX_LENGTH = 100

# Model
MODEL_PATH = 'api/model/pretrained'