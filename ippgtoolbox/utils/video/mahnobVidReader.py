"""
-------------------------------------------------------------------------------
Created: 27.01.2022, 13:45
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the refactored MAHNOB-HCI database. It expects a video to be split in frames that lie in a directory.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import numpy as np
import cv2
import re
from .baseVidReaders import VidFramesReader


class MAHNOBVidReader(VidFramesReader):
    """Video reader class for videos of the refactored MAHNOB-HCI database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """

    def __init__(self, absFileName):
        super().__init__(absFileName, 'png')

    def _getMetaData(self):
        # from the session.xml files (the manual says 60 fps; cv2.VideoCapture says 61 fps)
        super()._getMetaData(60.9708, (8, 8, 8))
