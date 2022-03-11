"""
-------------------------------------------------------------------------------
Created: 16.02.2022, 09:50
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Reader class to read the video files from MR-NIRP-Indoor dataset. This dataset consists of .pgm files where demosaicing must be applied.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import cv2
from .baseVidReaders import VidFramesReader


class MRNIRPINDOORVidReader(VidFramesReader):
    """Video reader class for videos of the MR-NIRP-Indoor database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """

    def __init__(self, absFileName, bit_depth=10):
        super().__init__(absFileName, 'pgm')
        self.bit_depth = bit_depth

    def getFrameAtIndex(self, frameIndex):
        self._checkFrameIndex(frameIndex)
        abs_frame_path = os.path.join(
            self.absFileName, self.frame_names[frameIndex])
        # read the images in as 16bit unsigned integers
        frame = cv2.imread(abs_frame_path, cv2.IMREAD_ANYDEPTH)
        # apply demosaicing
        if 'Subject6' in self.absFileName:
            # sensor format: 'bggr'
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        else:
            # sensor format: 'rggb'
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2RGB)
        # convert to specified bitdepth original pixel depth
        # original bit depth is 10bits but orientation is msb when read with cv
        frame = frame/2**6  # convert to 10bits
        if self.bit_depth == 8:
            frame = frame/(2**10-1) * (2**8-1)
            frame = frame.astype('uint8')
        elif self.bit_depth == 10:
            frame = frame.astype('uint16')
        else:
            ValueError('The specified bit depth is not implemented.')
        return frame

    def _getMetaData(self):
        # 30fps from paper containing dataset description
        bd = self.bit_depth
        super()._getMetaData(30, (bd, bd, bd))
