"""
-------------------------------------------------------------------------------
Created: 16.02.2022, 10:25
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Purpose: Tests for the MRNIRPINDOORVidReader class.
-------------------------------------------------------------------------------
"""
# %%
from ippgtoolbox.utils import MRNIRPINDOORVidReader
import matplotlib.pyplot as plt

VIDEO_PATH = '/media/super_fast_storage_2/matthieu_scherpf/2018_01_MR-NIRP-Indoor/measurements/Subject6/Subject6_motion_940/RGB/'


class TestMRNIRPINDOORVidReader:

    def __init__(self):
        self.vid_obj = MRNIRPINDOORVidReader(VIDEO_PATH)

    def test_readMetaData(self):
        self.vid_obj.readMetaData()

    def test_reader_next(self):
        self.frames = []
        for i, frame in enumerate(self.vid_obj.reader_next()):
            self.frames.append(frame/(2**self.vid_obj.bitsPerChannel[0]-1))
            if i == 10:
                break
        self.frames_startFrameIndex = []
        for i, frame in enumerate(self.vid_obj.reader_next(
                startFrameIndex=100)):
            self.frames_startFrameIndex.append(
                frame/(2**self.vid_obj.bitsPerChannel[0]-1))
            if i == 10:
                break
        plt.figure()
        plt.imshow(self.frames[0])
        plt.figure()
        plt.imshow(self.frames[-1])
        plt.figure()
        plt.imshow(self.frames_startFrameIndex[0])
        plt.figure()
        plt.imshow(self.frames_startFrameIndex[0])

    def test_reader_nextAtIndex(self):
        indexes = [0, 10, 100, 110]
        self.frames_atIndex = []
        for frame in self.vid_obj.reader_nextAtIndex(indexes):
            self.frames_atIndex.append(
                frame/(2**self.vid_obj.bitsPerChannel[0]-1))
        assert (self.frames_atIndex[0]-self.frames[0]).sum() == 0, 'True'
        assert (self.frames_atIndex[1]-self.frames[-1]).sum() == 0, 'True'
        assert (
            self.frames_atIndex[2]-self.frames_startFrameIndex[0]).sum() == 0, 'True'
        assert (
            self.frames_atIndex[3]-self.frames_startFrameIndex[-1]).sum() == 0, 'True'


if __name__ == '__main__':

    testMRNIRPINDOORVidReader = TestMRNIRPINDOORVidReader()
    testMRNIRPINDOORVidReader.test_readMetaData()
    testMRNIRPINDOORVidReader.test_reader_next()
    testMRNIRPINDOORVidReader.test_reader_nextAtIndex()

# %%
