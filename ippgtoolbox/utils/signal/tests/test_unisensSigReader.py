"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 20:09
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Tests for the UnisensSigReader class.
-------------------------------------------------------------------------------
"""
# %%
from ippgtoolbox.utils import UnisensSigReader
import matplotlib.pyplot as plt


FILE_PATH = '/media/fast_storage/matthieu_scherpf/2016_04_ColdStressStudy/measurements/104_S2_unisens/PPG.bin'


class TestUnisensSigReader:

    def __init__(self):
        self.sig_obj = UnisensSigReader(FILE_PATH)

    def test_readMetaData(self):
        self.sig_obj.readMetaData()

    def test_readAllValues(self):
        vals = self.sig_obj.readAllValues()
        plt.plot(vals[int(0.5*1e6):int(0.5*1e6+2000)])


if __name__ == '__main__':

    testUnisensSigReader = TestUnisensSigReader()
    testUnisensSigReader.test_readMetaData()
    testUnisensSigReader.test_readAllValues()

# %%
