"""
Some constants to configure
"""

# Preprocessing
FILTER_WORDS = ['removed', 'deleted', 'tl;dr']  # filter out comments containing these words
RAW_DATA_PATH = 'data/raw'

# Dodel
MODEL_PATH = 'pretrained'     # path where model (checkpoints) is saved

# OpenNMT generator
ONMT_MODEL_NAME = "hot-take-model"
ONMT_SENTENCEPIECE_MODEL = "hot-take-model/1/assets/wmtende.model"
ONMT_TIMEOUT = 5
ONMT_MAX_LENGTH = 100