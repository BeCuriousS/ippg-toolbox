"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 18:39
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the BP4D+ database. It expects a video to be split in frames that lie in a directory.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import numpy as np
import cv2
import re
from .baseVidReaders import VidFramesReader


class BP4DVidReader(VidFramesReader):
    """Video reader class for videos of the BP4D+ database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """

    def __init__(self, absFileName, **kwargs):
        super().__init__(absFileName, 'jpg', **kwargs)

    def _getMetaData(self):
        super()._getMetaData(25, (8, 8, 8))
