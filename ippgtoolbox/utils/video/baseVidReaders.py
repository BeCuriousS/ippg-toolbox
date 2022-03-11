"""
-------------------------------------------------------------------------------
Created: 16.02.2022, 16:51
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Base classes for the several video readers, i.e. image-based (e.g. *.or file-based (e.g. .avi).
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import cv2
import numpy as np
import re


class VidFileReader:
    """Base class for reading from official video files (e.g. .avi).
    """

    def __init__(self, absFileName):
        self.absFileName = absFileName

    def reader_next(self, startFrameIndex=0, return_ts=False):
        """Generator that yields frames and timestamps frame by frame until the video is completely read.

        Args:
            startFrameIndex (int, optional): frame index from which to start reading
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False. For compatibility with other video reader classes as timestamps are always None.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        capObj = cv2.VideoCapture(self.absFileName)
        self._getMetaData(capObj)
        if startFrameIndex != 0:
            capObj.set(cv2.CAP_PROP_POS_FRAMES, startFrameIndex)
        self._frameCounter = startFrameIndex
        while self._frameCounter < self.numberOfFrames:
            next_timestamp = self._getNextTimestamp(capObj)
            next_frame = self._getNextFrame(capObj)
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
        capObj = cv2.VideoCapture(self.absFileName)
        self._getMetaData(capObj)
        if return_ts:
            for idx in frameIndexes:
                next_timestamp = None
                next_frame = self.getFrameAtIndex(capObj, idx)
                yield (next_frame, next_timestamp)
        else:
            for idx in frameIndexes:
                next_frame = self.getFrameAtIndex(capObj, idx)
                yield next_frame

    def readMetaData(self):
        capObj = cv2.VideoCapture(self.absFileName)
        self._getMetaData(capObj)

    def getFrameAtIndex(self, capObj: 'cv2.VideoCapture', frameIndex):
        self._checkFrameIndex(frameIndex)
        capObj.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        check, frame = capObj.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def getTimestampAtIndex(self, capObj: 'cv2.VideoCapture', frameIndex):
        self._checkFrameIndex(frameIndex)
        capObj.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        timestamp = capObj.get(cv2.CAP_PROP_POS_MSEC)  # position in millisecs.
        return timestamp*1e3  # convert to microsecs

    def getAllTimestamps(self, capObj: 'cv2.VideoCapture'):
        raise NotImplementedError(
            "This function does not exist for this video type.")

    def _getMetaData(self, capObj):
        self.numberOfFrames = capObj.get(cv2.CAP_PROP_FRAME_COUNT)
        self.width = int(capObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (self.height, self.width)
        self.fileSize = os.path.getsize(self.absFileName)
        self.framesPerSecond = capObj.get(cv2.CAP_PROP_FPS)
        self.duration = self.numberOfFrames / self.framesPerSecond
        self.colorModel = 'RGB'
        self.bitsPerChannel = (8, 8, 8)

    def _getNextFrame(self, capObj: 'cv2.VideoCapture'):
        check, frame = capObj.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _getNextTimestamp(self, capObj: 'cv2.VideoCapture'):
        timestamp = capObj.get(cv2.CAP_PROP_POS_MSEC)  # position in millisecs.
        return timestamp*1e3  # convert to microsecs

    def _checkFrameIndex(self, frameIndex):
        valid = False
        if (frameIndex < 0) or (frameIndex % 1) != 0:
            Logger.logError('frame index must be a positive integer.')
            raise ValueError('frameindex value out of range')
        elif frameIndex >= self.numberOfFrames:
            Logger.logError('frame index larger than number of frames.')
            raise ValueError('frameindex value out of range')
        else:
            valid = True
        if not valid:
            raise ValueError('Invalid frameindex')
        return valid

    def _resetPointer(self, capObj: 'cv2.VideoCapture'):
        capObj.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frameCounter = 0


class VidFramesReader:
    """Base class for reading from videos consisting of frames (e.g. .png, .jpg, .pgm, etc.).
    """

    def __init__(self, absFileName, img_type):
        self.absFileName = absFileName
        self.img_type = img_type
        self.frame_names = []

    def reader_next(self, startFrameIndex=0, return_ts=False):
        """Generator that yields frames and timestamps frame by frame until the video is completely read.

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
        raise NotImplementedError(
            "This function does not exist for this video type.")

    def getAllTimestamps(self, frameIndex):
        raise NotImplementedError(
            "This function does not exist for this video type.")

    def _getMetaData(self, framesPerSecond, bitsPerChannel):
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
        self.duration = self.numberOfFrames / framesPerSecond
        self.colorModel = 'RGB'
        self.framesPerSecond = framesPerSecond
        self.bitsPerChannel = bitsPerChannel

    def _getNextFrame(self):
        raise NotImplementedError(
            "This function does not exist for this video type.")

    def _getNextTimestamp(self):
        raise NotImplementedError(
            "This function does not exist for this video type.")

    def _checkFrameIndex(self, frameIndex):
        valid = False
        if (frameIndex < 0) or (frameIndex % 1) != 0:
            Logger.logError('frame index must be a positive integer.')
            raise ValueError('frameindex value out of range')
        elif frameIndex >= self.numberOfFrames:
            Logger.logError('frame index larger than number of frames.')
            raise ValueError('frameindex value out of range')
        else:
            valid = True
        if not valid:
            raise ValueError('Invalid frameindex')
        return valid

    def _resetPointer(self):
        self._frameCounter = 0
