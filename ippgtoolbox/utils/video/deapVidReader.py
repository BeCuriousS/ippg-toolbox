"""
-------------------------------------------------------------------------------
Created: 03.03.2022, 19:59
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the DEAP database. It is based on the standard reader class of openCV (cv2.Videoreader).
-------------------------------------------------------------------------------
"""
from ..logger import Logger
from .baseVidReaders import VidFileReader


class DEAPVidReader(VidFileReader):
    """Video reader class for videos of the DEAP database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """
