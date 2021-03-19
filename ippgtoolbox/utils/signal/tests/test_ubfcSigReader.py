"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 20:09
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Tests for the UBFCSigReader class.
-------------------------------------------------------------------------------
"""
# %%
from utils import UBFCSigReader
import matplotlib.pyplot as plt


FILE_PATH = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/measurements/subject1/ground_truth.txt'


class TestUBFCSigReader:

    def __init__(self):
        self.sig_obj = UBFCSigReader(FILE_PATH)

    def test_readMetaData(self):
        self.sig_obj.readMetaData()

    def test_readAllValues(self):
        vals = self.sig_obj.readAllValues()
        plt.plot(vals)


if __name__ == '__main__':

    testUBFCSigReader = TestUBFCSigReader()
    testUBFCSigReader.test_readMetaData()
    testUBFCSigReader.test_readAllValues()

# %%
