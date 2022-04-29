"""
-------------------------------------------------------------------------------
Created: 16.02.2022, 16:45
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the VIPL_HR_V1 database. It is based on the standard reader class of openCV (cv2.Videoreader).
-------------------------------------------------------------------------------
"""
from .baseVidReaders import VidFileReader
from ..logger import Logger
import os
import cv2
import numpy as np
import pickle


class VIPLHRV1VidReader(VidFileReader):
    """Video reader class for videos of the VIPL_HR_V1 database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """

    def _getMetaData(self, capObj):
        super()._getMetaData(capObj)
        # NOTE the setting of this depends on which source for this database reader is chosen. There are 3 source devices for RGB videos.
        vid_ts_file = os.path.join(os.path.split(
            self.absFileName)[0], 'time.txt')
        # load and convert from milsec to microsec
        self.timestamps = (np.loadtxt(vid_ts_file)*1e3).astype(np.int64)

    def _getNextTimestamp(self, capObj: 'cv2.VideoCapture'):
        # get position of frame
        frame_pos = int(capObj.get(cv2.CAP_PROP_POS_FRAMES))
        return self.timestamps[frame_pos-1]
