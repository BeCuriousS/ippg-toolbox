"""
-------------------------------------------------------------------------------
Created: 07.02.2022, 12:48
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the PURE dataset.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import numpy as np
import cv2
import re
from .baseVidReaders import VidFramesReader


class PUREVidReader(VidFramesReader):
    """Video reader class for videos of the PURE database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """

    def __init__(self, absFileName):
        super().__init__(absFileName, 'png')

    def _getMetaData(self):
        # from db manual/publication: Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
        super()._getMetaData(30, (8, 8, 8))
