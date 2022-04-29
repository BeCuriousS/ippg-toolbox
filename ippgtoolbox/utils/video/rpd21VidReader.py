"""
-------------------------------------------------------------------------------
Created: 04.03.2022, 11:29
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the "Remote Pulse Detection 21" (RPD21) database. It is based on the standard reader class of openCV (cv2.Videoreader).
-------------------------------------------------------------------------------
"""
from ..logger import Logger
from .baseVidReaders import VidFileReader
import os
import numpy as np
import pickle
import cv2


class RPD21VidReader(VidFileReader):
    """Video reader class for videos of the "Remote Pulse Detection 21" (RPD21) database. This class is for simple compatibility with the MSRReader/MSR2Reader class.

    NOTE The timestamps are computed using a fixed rate of framesPerSecond. Eventhough the timestamps could be read from the .mp4 container there occur errors at the end resulting in timestamps being zero. The origin of this error is unclear but must lie somewhere within opencv.
    """

    def _getMetaData(self, capObj):
        super()._getMetaData(capObj)
        # self.numberOfFrames = capObj.get(cv2.CAP_PROP_FRAME_COUNT) # not working here as there are some videos with one frame more or less due to some error (i.e. erroneous frames)
        nof_file = os.path.join(os.path.split(
            self.absFileName)[0], 'vid_nof_custom.p')
        with open(nof_file, 'rb') as f:
            self.numberOfFrames = pickle.load(f)
        self.duration = self.numberOfFrames / self.framesPerSecond
        timestamps = np.arange(0, self.numberOfFrames) * \
            1/self.framesPerSecond * 1e6
        self.timestamps = timestamps.astype(np.int64)

    def _getNextTimestamp(self, capObj: 'cv2.VideoCapture'):
        # get position of frame
        frame_pos = int(capObj.get(cv2.CAP_PROP_POS_FRAMES))
        return self.timestamps[frame_pos-1]
