"""
ariannamethod — notorch + Chuck Optimizer (VLM edition)

C line:  notorch.c / notorch.h (pure C neural framework)
         notorch_vision.h (image loading, transforms, patch extraction)
         stb_image.h (JPEG/PNG/BMP decoder)
Python:  notorch_nn.py (ctypes binding to libnotorch)
         chuck.py (self-aware optimizer via ctypes)

no torch. no pip install torch. no 2.7 GB.
"""

from .notorch_nn import *
from .chuck import ChuckOptimizer

__all__ = ['Tensor', 'Parameter', 'Module', 'Linear', 'Embedding',
           'RMSNorm', 'NotorchEngine', 'softmax', 'silu', 'cross_entropy',
           'multinomial', 'seed', 'ChuckOptimizer',
           '_lib', '_get_tensor_struct', '_NtTapeEntry', '_NtTensor']
