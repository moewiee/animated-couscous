from yacs.config import CfgNode as CN


_C = CN()

_C.EXP = "" # Experiment name
_C.DEBUG = False

_C.SYSTEM = CN()
_C.SYSTEM.SEED = 0
_C.SYSTEM.FP16 = True
_C.SYSTEM.OPT_L = "O2"
_C.SYSTEM.CUDA = True
_C.SYSTEM.NUM_WORKERS = 8

_C.DIRS = CN()
_C.DIRS.TRAIN_IMAGES = ""
_C.DIRS.VALIDATION_IMAGES = ""
_C.DIRS.TEST_IMAGES = ""
_C.DIRS.WEIGHTS = "./weights/"
_C.DIRS.OUTPUTS = "./outputs/"
_C.DIRS.LOGS = "./logs/"

_C.DATA = CN()
_C.DATA.TRAIN_CSV_FILE = ""
_C.DATA.VALIDATION_CSV_FILE = ""
_C.DATA.AUGMENT = "randaug" # dont use augmix
_C.DATA.RANDAUG = CN()
_C.DATA.RANDAUG.N = 3
_C.DATA.RANDAUG.M = 11
_C.DATA.AUGMIX = CN()
_C.DATA.AUGMIX.ALPHA = 1.
_C.DATA.AUGMIX.BETA = 1.

_C.DATA.CUTMIX = False
_C.DATA.MIXUP = False
_C.DATA.GRIDMASK = False
_C.DATA.CM_ALPHA = 1.0
_C.DATA.IMG_SIZE = 224
_C.DATA.INP_CHANNEL = 3

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 40
_C.TRAIN.NUM_CYCLES = 4
_C.TRAIN.BATCH_SIZE = 8

_C.INFER = CN()
_C.INFER.TTA = False

_C.OPT = CN()
_C.OPT.OPTIMIZER = "adamw"
_C.OPT.GD_STEPS = 1
_C.OPT.WARMUP_EPOCHS = 4
_C.OPT.BASE_LR = 1e-3
_C.OPT.WEIGHT_DECAY = 1e-2
_C.OPT.WEIGHT_DECAY_BIAS = 0.0
_C.OPT.EPSILON = 1e-3
_C.OPT.SWA = CN()
_C.OPT.SWA.START = 10
_C.OPT.SWA.FREQ = 5
_C.OPT.SWA.THRESHOLD = 1.0

_C.LOSS = CN()
_C.LOSS.NAME = "ce"
_C.LOSS.GAMMA = 2.

_C.MODEL = CN()
_C.MODEL.NAME = "resnet50"
_C.MODEL.PRETRAINED = True
_C.MODEL.ACTIVATION = "relu"
_C.MODEL.POOL_TYPE = "avg" # Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax'
_C.MODEL.DROPOUT = 0.3
_C.MODEL.DROP_CONNECT = 0.2
_C.MODEL.CLS_HEAD = 'linear'
_C.MODEL.NUM_CLASSES = 2

_C.CONST = CN()


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`