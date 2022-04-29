"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 18:39
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the UBFC-RPPG database. It is based on the standard reader class of openCV (cv2.Videoreader).
-------------------------------------------------------------------------------
"""
from ..logger import Logger
from .baseVidReaders import VidFileReader
import cv2
import os
import pickle


class UBFCRPPGVidReader(VidFileReader):
    """Video reader class for videos of the UBFC-RPPG database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """

    def _getMetaData(self, capObj):
        super()._getMetaData(capObj)
        vid_ts_file = os.path.join(os.path.split(
            self.absFileName)[0], 'video_timestamps.p')
        with open(vid_ts_file, 'rb') as f:
            self.timestamps = pickle.load(f)

    def _getNextTimestamp(self, capObj: 'cv2.VideoCapture'):
        # get position of frame
        frame_pos = int(capObj.get(cv2.CAP_PROP_POS_FRAMES))
        return self.timestamps[frame_pos-1]
