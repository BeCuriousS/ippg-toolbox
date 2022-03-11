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


class VICARPPG2VidReader(VidFileReader):
    """Video reader class for videos of the VICARPPG-2 database. This class is for simple compatibility with the MSRReader/MSR2Reader class.
    """
