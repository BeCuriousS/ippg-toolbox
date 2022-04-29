"""
-------------------------------------------------------------------------------
Created: 16.02.2022, 16:45
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the VICARPPG-2 database. It is based on the standard reader class of openCV (cv2.Videoreader).
-------------------------------------------------------------------------------
"""
from .baseVidReaders import VidFileReader
from ..logger import Logger
import os
import cv2
import numpy as np


class VICARPPG2VidReader(VidFileReader):
    """Video reader class for videos of the VICARPPG-2 database. This class is for simple compatibility with the MSRReader/MSR2Reader class.

    NOTE The timestamps are computed using a fixed rate of framesPerSecond. Eventhough the timestamps could be read from the .mp4 container there occur errors at the end resulting in timestamps being zero. The origin of this error is unclear but must lie somewhere within opencv.
    """

    def _getMetaData(self, capObj):
        super()._getMetaData(capObj)
        timestamps = np.arange(0, self.numberOfFrames) * \
            1/self.framesPerSecond * 1e6
        self.timestamps = timestamps.astype(np.int64)

    def _getNextTimestamp(self, capObj: 'cv2.VideoCapture'):
        # get position of frame
        frame_pos = int(capObj.get(cv2.CAP_PROP_POS_FRAMES))
        return self.timestamps[frame_pos-1]
