"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 20:06
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Signal reader class for the BP4D+ database
-------------------------------------------------------------------------------
"""
import numpy as np


class BP4DSigReader():
    """
    This class is built similar to the UnisensReader but not inherited from because it differs too much.
    """

    def __init__(self, full_file_name):
        self.full_file_name = full_file_name

    def readAllValues(self):
        self._getMetaData()
        return np.loadtxt(self.full_file_name)

    def readMetaData(self):
        self._getMetaData()

    def _getMetaData(self):
        self.sampleRate = 1000
