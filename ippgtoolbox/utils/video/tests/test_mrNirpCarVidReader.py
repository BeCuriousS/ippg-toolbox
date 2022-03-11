"""
-------------------------------------------------------------------------------
Created: 15.02.2022, 23:58
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Purpose: Tests for the MRNIRPCARVidReader class.
-------------------------------------------------------------------------------
"""
# %%
from ippgtoolbox.utils import MRNIRPCARVidReader
import matplotlib.pyplot as plt

VIDEO_PATH = '/media/super_fast_storage_2/matthieu_scherpf/2020_01_MR-NIRP-Car/measurements/Subject1/subject1_driving_large_motion_975/RGB/'


class TestMRNIRPCARVidReader:

    def __init__(self):
        self.vid_obj = MRNIRPCARVidReader(VIDEO_PATH)
        self.vid_obj_8bit = MRNIRPCARVidReader(VIDEO_PATH, bit_depth=8)

    def test_readMetaData(self):
        self.vid_obj.readMetaData()

    def test_reader_next(self):
        self.frames = []
        self.frames_8bit = []
        for i, frame in enumerate(self.vid_obj.reader_next()):
            self.frames.append(frame/(2**self.vid_obj.bitsPerChannel[0]-1))
            if i == 10:
                break
        for i, frame in enumerate(self.vid_obj_8bit.reader_next()):
            self.frames_8bit.append(frame)
            if i == 10:
                break
        self.frames_startFrameIndex = []
        self.frames_startFrameIndex_8bit = []
        for i, frame in enumerate(self.vid_obj.reader_next(
                startFrameIndex=100)):
            self.frames_startFrameIndex.append(
                frame/(2**self.vid_obj.bitsPerChannel[0]-1))
            if i == 10:
                break
        for i, frame in enumerate(self.vid_obj_8bit.reader_next(
                startFrameIndex=100)):
            self.frames_startFrameIndex_8bit.append(frame)
            if i == 10:
                break
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.frames[0])
        axs[1].imshow(self.frames_8bit[0])
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.frames[-1])
        axs[1].imshow(self.frames_8bit[-1])
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.frames_startFrameIndex[0])
        axs[1].imshow(self.frames_startFrameIndex_8bit[0])
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.frames_startFrameIndex[-1])
        axs[1].imshow(self.frames_startFrameIndex_8bit[-1])

    def test_reader_nextAtIndex(self):
        indexes = [0, 10, 100, 110]
        self.frames_atIndex = []
        for frame in self.vid_obj.reader_nextAtIndex(indexes):
            self.frames_atIndex.append(
                frame/(2**self.vid_obj.bitsPerChannel[0]-1))
        self.frames_atIndex_8bit = []
        for frame in self.vid_obj_8bit.reader_nextAtIndex(indexes):
            self.frames_atIndex_8bit.append(frame)
        assert (self.frames_atIndex[0]-self.frames[0]).sum() == 0, 'True'
        assert (self.frames_atIndex[1]-self.frames[-1]).sum() == 0, 'True'
        assert (
            self.frames_atIndex[2]-self.frames_startFrameIndex[0]).sum() == 0, 'True'
        assert (
            self.frames_atIndex[3]-self.frames_startFrameIndex[-1]).sum() == 0, 'True'
        assert (self.frames_atIndex_8bit[0] -
                self.frames_8bit[0]).sum() == 0, 'True'
        assert (self.frames_atIndex_8bit[1] -
                self.frames_8bit[-1]).sum() == 0, 'True'
        assert (
            self.frames_atIndex_8bit[2]-self.frames_startFrameIndex_8bit[0]).sum() == 0, 'True'
        assert (
            self.frames_atIndex_8bit[3]-self.frames_startFrameIndex_8bit[-1]).sum() == 0, 'True'


if __name__ == '__main__':

    testMRNIRPCARVidReader = TestMRNIRPCARVidReader()
    testMRNIRPCARVidReader.test_readMetaData()
    testMRNIRPCARVidReader.test_reader_next()
    testMRNIRPCARVidReader.test_reader_nextAtIndex()

# %%
