"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 20:09
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Tests for the BP4DSigReader class.
-------------------------------------------------------------------------------
"""
# %%
from utils import BP4DSigReader
import matplotlib.pyplot as plt


FILE_PATH = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/measurements/F001_T1/signals/Pulse Rate_BPM.txt'


class TestBP4DSigReader:

    def __init__(self):
        self.sig_obj = BP4DSigReader(FILE_PATH)

    def test_readMetaData(self):
        self.sig_obj.readMetaData()

    def test_readAllValues(self):
        vals = self.sig_obj.readAllValues()
        plt.plot(vals)


if __name__ == '__main__':

    testBP4DSigReader = TestBP4DSigReader()
    testBP4DSigReader.test_readMetaData()
    testBP4DSigReader.test_readAllValues()

# %%
