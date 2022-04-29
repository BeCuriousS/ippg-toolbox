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

    def __init__(self, absFileName, return_type='uint16', **kwargs):
        super().__init__(absFileName, 'pgm', return_type=return_type, **kwargs)

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
        # original bit depth is 10bits. When read with opencv the values starting from msb are used. Therefore division by (2**16-1) instead of (2**10-1). Returns float with range [0., 1.]. Conversion is then done in class method reader_next/reader_nextAtIndex.
        frame = frame/(2**16-1)
        return frame

    def _getMetaData(self):
        # 30fps from paper containing dataset description
        super()._getMetaData(30, (10, 10, 10))
