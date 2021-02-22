"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 20:05
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Signal reader class for the UBFC database.
-------------------------------------------------------------------------------
"""
import numpy as np


class UBFCSigReader():
    """
    This class is built similar to the UnisensReader but not inherited from because it differs too much.
    """

    def __init__(self, full_file_name, row=0):
        self.full_file_name = full_file_name
        self.dataRow = row

    def readAllValues(self):
        self._getMetaData()
        if self.dataRow == 2:
            # timestamps from seconds to microseconds (as for unisens databases)
            return np.squeeze(np.loadtxt(self.full_file_name)[self.dataRow]) * 10**6
        return np.squeeze(np.loadtxt(self.full_file_name)[self.dataRow])

    def readMetaData(self):
        # in the following steps if sample rate equals None it is assumed, that there is one sample point for each frame
        self._getMetaData()

    def _getMetaData(self):
        self.sampleRate = None
