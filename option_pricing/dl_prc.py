import cupy
import numpy as np
import math
import time
import torch
cupy.cuda.set_allocator(None)
from torch.utils.dlpack import from_dlpack


#############
# 基于深度学习的障碍期权定价的加速方式
#############

