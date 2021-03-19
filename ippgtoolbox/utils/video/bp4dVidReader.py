"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 18:39
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader implemented for the BP4D+ database. It expects a video to be split in frames that lie in a directory.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import numpy as np
import cv2
import re
from .ubfcVidReader import UBFCVidReader


class BP4DVidReader(UBFCVidReader):
    """Video reader class for videos of the BP4D+ database. This class is for simple compatibility with the MSRReader/MSR2Reader class. It is ment for databases where the video is split into frames.
    """

    def __init__(self, absFileName, img_type='jpg'):
        super().__init__(absFileName)
        self.img_type = img_type
        self.frame_names = []

    def reader_next(self, startFrameIndex=0, return_ts=False):
        """Generator that yields frames and timestamps frame by frame until the video is completly read.

        Args:
            startFrameIndex (int, optional): frame index from which to start reading
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False. For compatibility with other video reader classes as timestamps are always None.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        self._getMetaData()
        self._frameCounter = startFrameIndex
        while self._frameCounter < self.numberOfFrames:
            next_timestamp = None
            next_frame = self.getFrameAtIndex(self._frameCounter)
            self._frameCounter += 1
            if return_ts:
                yield (next_frame, next_timestamp)
            else:
                yield next_frame

    def reader_nextAtIndex(self, frameIndexes, return_ts=False):
        """
        Generator that yields frames and timestamps at given frame indexes.

        Args:
            frameIndexes (iterable): Listlike object containing the indexes of the frames of interest in the video
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False. For compatibility with other video reader classes as timestamps are always None.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        self._getMetaData()
        if return_ts:
            for idx in frameIndexes:
                next_timestamp = None
                next_frame = self.getFrameAtIndex(idx)
                yield (next_frame, next_timestamp)
        else:
            for idx in frameIndexes:
                next_frame = self.getFrameAtIndex(idx)
                yield next_frame

    def readMetaData(self):
        self._getMetaData()

    def getFrameAtIndex(self, frameIndex):
        self._checkFrameIndex(frameIndex)
        abs_frame_path = os.path.join(
            self.absFileName, self.frame_names[frameIndex])
        frame = cv2.imread(abs_frame_path)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def getTimestampAtIndex(self, frameIndex):
        super().getTimestampAtIndex(None, frameIndex)

    def getAllTimestamps(self, frameIndex):
        super().getAllTimestamps(None)

    def _getMetaData(self):
        self.frame_names = sorted([fn for fn in os.listdir(
            self.absFileName) if re.search('.'+self.img_type, fn, re.I) != None])
        self.numberOfFrames = len(self.frame_names)
        expl_frame = cv2.imread(os.path.join(
            self.absFileName, self.frame_names[0]))
        self.height = expl_frame.shape[0]
        self.width = expl_frame.shape[1]
        self.resolution = (self.height, self.width)
        self.fileSize = np.sum([os.path.getsize(os.path.join(
            self.absFileName, fn)) for fn in self.frame_names])
        self.framesPerSecond = 25
        self.duration = self.numberOfFrames / self.framesPerSecond
        self.colorModel = 'RGB'
        self.bitsPerChannel = (8, 8, 8)

    def _getNextFrame(self):
        raise NotImplementedError(
            "This function does not exist for this video type.")

    def _getNextTimestamp(self):
        super()._getNextTimestamp(None)

    def _checkFrameIndex(self, frameIndex):
        super()._checkFrameIndex(frameIndex)

    def _resetPointer(self):
        self._frameCounter = 0
