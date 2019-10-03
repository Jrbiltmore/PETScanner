from yacs.config import CfgNode as CN


_C = CN()

_C.NAME = "Validation 1"
_C.OUTPUT_DIR = "/data/PETScanner/results_1"
_C.MODEL_SAVE_FILENAME = "model.pkl"

_C.DATASET = CN()
_C.DATASET.PATH = '/data/PETScanner/dataset_1.pkl'

_C.MODEL = CN()

_C.MODEL.CHANNELS_START = 64

_C.TRAIN = CN()

_C.TRAIN.EPOCHS = 150
_C.TRAIN.BASE_LEARNING_RATE = 1e-3
_C.TRAIN.LEARNING_DECAY_RATE = 0.1
_C.TRAIN.LEARNING_DECAY_STEPS = 50


def get_cfg_defaults():
    return _C.clone()
