from typing import List

import torch
from enum import Enum
from torch.nn import ReLU, ReLU6, Hardswish, Hardsigmoid
from torch.nn import BatchNorm2d
from torchvision.ops import FrozenBatchNorm2d

IMAGENET_TRAIN_DIR = 'ILSVRC2012_img_train'
IMAGENET_VAL_DIR = 'ILSVRC2012_img_val_TFrecords'

NORMALIZATION_IMAGENET_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_IMAGES = 1300
BATCH_AXIS, CHANNEL_AXIS, H_AXIS, W_AXIS = 0, 1, 2, 3

IMAGE_INPUT = 'image_input'
TORCH_ACTIVATION_FUNCTIONS = (ReLU, ReLU6, Hardswish, Hardsigmoid)
BATCHNORM_PYTORCH_LAYERS = (BatchNorm2d, FrozenBatchNorm2d)
RESNET18 = 'resnet18'
RESNET50 = 'resnet50'
MBV2 = 'mobilenet_v2'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EnumBaseClass(Enum):
    """
    Base class for Enums with additional utility methods.
    """
    @classmethod
    def get_values(cls) -> List[str]:
        """
        Returns a list of all Enum member values.
        """
        return [mem.value for mem, mem in cls.__members__.items()]

class DistillDataAlg(EnumBaseClass):
    """
    Enumeration of data distillation algorithms.
    """
    RANDOM_NOISE = 'random_noise'
    DGH = 'dgh'
    REAL_DATA = 'real_data'

class LayerWeightingType(EnumBaseClass):
    """
    Enumeration of layer weighting types.
    """
    AVERAGE = 'average'



class BatchNormAlignemntLossType(EnumBaseClass):
    """
    Enumeration of batch normalization alignment loss types.
    """
    L2_SQUARE = 'l2_square'

class SchedularType(EnumBaseClass):
    """
    Enumeration of scheduler types.
    """
    REDUCE_ON_PLATEAU = 'reduce_on_plateau'
    REDUCE_ON_PLATEAU_WITH_RESET = 'reduce_on_plateau_with_reset'
    STEP = 'step'
    CONST = 'const'