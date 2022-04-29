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


class TestMRNIRPCARVidReader:

    def __init__(self):
        self.vid_obj = MRNIRPINDOORVidReader(
            VIDEO_PATH)
        self.vid_obj_8bit = MRNIRPINDOORVidReader(
            VIDEO_PATH, return_type='uint8')
        self.vid_obj_float32 = MRNIRPINDOORVidReader(
            VIDEO_PATH, return_type='float32')
        self.vid_obj_float64 = MRNIRPINDOORVidReader(
            VIDEO_PATH, return_type='float64')

    def test_readMetaData(self):
        self.vid_obj.readMetaData()

    def test_reader_next(self):
        self.frames = []
        self.frames_8bit = []
        self.frames_startFrameIndex = []
        self.frames_startFrameIndex_8bit = []
        for i, frame in enumerate(self.vid_obj.reader_next()):
            self.frames.append(
                frame/(2**self.vid_obj.bitsPerChannel[0]-1))
            if i == 10:
                break
        for i, frame in enumerate(self.vid_obj_8bit.reader_next()):
            self.frames_8bit.append(frame)
            if i == 10:
                break
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

        print(self.frames[-1].min(), self.frames[0].max())
        print(self.frames_8bit[-1].min(), self.frames_8bit[0].max())
        print(self.frames_startFrameIndex[-1].min(),
              self.frames_startFrameIndex[0].max())
        print(self.frames_startFrameIndex_8bit[-1].min(),
              self.frames_startFrameIndex_8bit[0].max())

    def test_frame_return_type_ranges(self):
        fig, axs = plt.subplots(1, 4)
        for i, frame in enumerate(self.vid_obj.reader_next()):
            axs[0].imshow(frame)
            print('uint16:', frame.min(), frame.max())
            break
        for i, frame in enumerate(self.vid_obj_8bit.reader_next()):
            axs[1].imshow(frame)
            print('uint8:', frame.min(), frame.max())
            break
        for i, frame in enumerate(self.vid_obj_float32.reader_next()):
            axs[2].imshow(frame)
            print('float32:', frame.min(), frame.max())
            break
        for i, frame in enumerate(self.vid_obj_float64.reader_next()):
            axs[3].imshow(frame)
            print('float64:', frame.min(), frame.max())
            break

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
    testMRNIRPCARVidReader.test_frame_return_type_ranges()

# %%
