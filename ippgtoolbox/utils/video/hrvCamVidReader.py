"""
-------------------------------------------------------------------------------
Created: 15.02.2022, 23:28
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Reader class to read the video files from HRVCam dataset. This dataset consists of .pgm files where demosaicing must be applied.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import numpy as np
import cv2
import re
from .baseVidReaders import VidFramesReader


class HRVCAMVidReader(VidFramesReader):
    """Video reader class for videos of the HRVCam database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """

    def __init__(self, absFileName, **kwargs):
        super().__init__(absFileName, 'pgm', **kwargs)

    def getFrameAtIndex(self, frameIndex):
        self._checkFrameIndex(frameIndex)
        abs_frame_path = os.path.join(
            self.absFileName, self.frame_names[frameIndex])
        # read the images in as 16bit unsigned integers
        frame = cv2.imread(abs_frame_path, cv2.IMREAD_ANYDEPTH)
        # apply demosaicing (sensor format: 'rggb')
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
        return frame

    def _getMetaData(self):
        # 30fps from paper containing dataset description
        super()._getMetaData(30, (8, 8, 8))
        absTsFileName = os.path.join(os.path.split(
            self.absFileName)[0], 'misc', 'CameraTimeLog0.txt')
        cam_ts = np.loadtxt(absTsFileName)
        # from secs to microsecs
        self.timestamps = (cam_ts*1e6).astype(np.int64)
