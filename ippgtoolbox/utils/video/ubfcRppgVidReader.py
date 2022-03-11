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


class UBFCRPPGVidReader(VidFileReader):
    """Video reader class for videos of the UBFC-RPPG database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """
