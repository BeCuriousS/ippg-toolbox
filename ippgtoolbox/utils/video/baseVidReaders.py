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
import warnings


def _img_type_conversion(img, bitdepth, new_type):

    max_val = (2**bitdepth-1)

    def _to_float():
        return (img/max_val)
    # ----------------------------------------------------------
    # Check image element types and convert accordingly
    # ----------------------------------------------------------
    assert new_type in ['uint8', 'uint16', 'float32',
                        'float64'], 'New type <<< {} >>> not yet supported.'.format(new_type)

    if img.dtype == 'uint8':
        if new_type == 'uint16':
            return img.astype(new_type)
        if new_type in ['float32', 'float64']:
            return _to_float().astype(new_type)

    if img.dtype == 'uint16':
        if new_type == 'uint8':
            return (_to_float()*255).astype(new_type)
        if new_type in ['float32', 'float64']:
            return _to_float().astype(new_type)

    if img.dtype in ['float32', 'float64']:
        if new_type == 'uint8':
            return (img*255).astype(new_type)
        if new_type == 'uint16':
            return (img*max_val).astype(new_type)
        else:
            return img.astype(new_type)
    # no conversion necessary
    return img


class VidFileReader:
    """Base class for reading from official video files (e.g. .avi).
    NOTE: There occur some uncertainties with the timestamps/fps computation using opencv. Be aware of this issue.
    """

    def __init__(self,
                 absFileName,
                 return_type='uint8'):
        """Creates a video reader class based on a video container file (e.g. .avi, .mp4).

        Args:
            absFileNmae (str): Absolute path to the video container file.
            return_type (str, optional): Set the array type to return. Can be 'uint8', 'uint16', 'float32' or 'float64'. If the original input range is adjusted if the dtype is to small, e.g. conversion of 10bit frame to uint8 (see following examples). Ranges depending on colordepth:
                - 8bit  -> dtype out -> uint8  with range [0, 255]
                - 10bit -> dtype out -> uint8  with range [0, 255]
                - 12bit -> dtype out -> uint8  with range [0, 255]
                - 8bit  -> dtype out -> uint16 with range [0, 255]
                - 10bit -> dtype out -> uint16 with range [0, 1023]
                - 12bit -> dtype out -> uint16 with range [0, 4095]
                - Any   -> dtype out -> float32 or float64 [0., 1.]
        """
        self.absFileName = absFileName
        self._return_type = return_type

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
        frame_counter = startFrameIndex
        while frame_counter < self.numberOfFrames:
            next_frame = self._getNextFrame(capObj)
            next_timestamp = self._getNextTimestamp(capObj)
            next_frame = _img_type_conversion(
                next_frame,
                self.bitsPerChannel[0],
                self._return_type)
            frame_counter += 1
            if return_ts:
                yield (next_frame, next_timestamp)
            else:
                yield next_frame

    def reader_nextAtIndex(self, frameIndexes,
                           return_ts=False,
                           use_opencv_set=True):
        """
        Generator that yields frames and timestamps at given frame indexes.
        NOTE timestamps can be buggy using this function. Reason is not clear but when searching the internet one can find information about opencv and timestamp extraction and its problems.
        NOTE it turns out that it's probably faster to use the .reader_next() method for video files (i.e. containers) and then only process the required frames at chosen index positions (this was observed behavior). This is why "use_opencv_set" was introduced as a parameter.

        Args:
            frameIndexes (iterable): Listlike object containing the indexes of the frames of interest in the video
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False. For compatibility with other video reader classes as timestamps are always None.
            use_opencv_set (bool, optional): Use opencv cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, idx) to extract frames. This can be very slow (maybe because of underlying compression). Read notes of this function for more information. Defaults to False.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        capObj = cv2.VideoCapture(self.absFileName)
        self._getMetaData(capObj)
        if use_opencv_set:
            for idx in frameIndexes:
                next_frame = self.getFrameAtIndex(capObj, idx)
                # position is already set correctly by previous method
                next_timestamp = self._getNextTimestamp(capObj)
                next_frame = _img_type_conversion(
                    next_frame,
                    self.bitsPerChannel[0],
                    self._return_type)
                if return_ts:
                    yield (next_frame, next_timestamp)
                else:
                    yield next_frame
        else:
            frame_counter = -1
            while frame_counter < self.numberOfFrames - 1:
                frame_counter += 1
                next_frame = self._getNextFrame(capObj)
                next_timestamp = self._getNextTimestamp(capObj)
                if frame_counter not in frameIndexes:
                    continue
                next_frame = _img_type_conversion(
                    next_frame,
                    self.bitsPerChannel[0],
                    self._return_type)
                if return_ts:
                    yield (next_frame, next_timestamp)
                else:
                    yield next_frame

    def readMetaData(self):
        # NOTE as for in some cases all the timestamps need to be read before any processing it makes sense to read them all at the beginning. To speed things up and to avoid reading the whole video it makes sense to read them once, save them to a file and then later read them from file. This is implemented in the child classes.
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
        return np.int64(timestamp*1e3)  # convert to microsecs

    def getAllTimestamps(self):
        try:
            return self.timestamps
        except AttributeError:
            raise AttributeError(
                "Before timestamps can be read, you must read the video meta data by calling the <readMetaData()> method.")

    def _getMetaData(self, capObj):
        self.numberOfFrames = int(capObj.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(capObj.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = (self.height, self.width)
        self.fileSize = os.path.getsize(self.absFileName)
        self.framesPerSecond = capObj.get(cv2.CAP_PROP_FPS)
        # NOTE seeking to the end of the video and reading out last timestamp seems more correct as there can occur missing frames. But this has to be done for each video reader seperately as the "CAP_PROP_FRAME_COUNT" property can have errors. The following duration calculation is only an estimate.
        self.duration = self.numberOfFrames / self.framesPerSecond
        self.colorModel = 'RGB'
        self.bitsPerChannel = (8, 8, 8)

    def _getNextFrame(self, capObj: 'cv2.VideoCapture'):
        check, frame = capObj.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _getNextTimestamp(self, capObj: 'cv2.VideoCapture'):
        timestamp = capObj.get(cv2.CAP_PROP_POS_MSEC)  # position in millisecs.
        return np.int64(timestamp*1e3)  # convert to microsecs

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


class VidFramesReader:
    """Base class for reading from videos consisting of frames (e.g. .png, .jpg, .pgm, etc.).
    """

    def __init__(self,
                 absFileName,
                 img_type,
                 return_type='uint8'):
        """Creates a video reader class based on a video consisting of frames (e.g. .jpg, .png, .pgm).

        Args:
            absFileName (str): Absolute path to the video folder containing the video frames.
            img_type (str): type of the video frames (e.g. .jpg, .png, .pgm).
            return_type (str, optional): Set the array type to return. Can be 'uint8', 'uint16', 'float32' or 'float64'. If the original input range is adjusted if the dtype is to small, e.g. conversion of 10bit frame to uint8 (see following examples). Ranges depending on colordepth:
                - 8bit  -> dtype out -> uint8  with range [0, 255]
                - 10bit -> dtype out -> uint8  with range [0, 255]
                - 12bit -> dtype out -> uint8  with range [0, 255]
                - 8bit  -> dtype out -> uint16 with range [0, 255]
                - 10bit -> dtype out -> uint16 with range [0, 1023]
                - 12bit -> dtype out -> uint16 with range [0, 4095]
                - Any   -> dtype out -> float32 or float64 [0., 1.]
        """
        self.absFileName = absFileName
        self.img_type = img_type
        self._return_type = return_type
        self.frame_names = []
        self.timestamps = []

    def reader_next(self, startFrameIndex=0, return_ts=False):
        """Generator that yields frames and timestamps frame by frame until the video is completely read.

        Args:
            startFrameIndex (int, optional): frame index from which to start reading
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False. For compatibility with other video reader classes as timestamps are always None.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        self._getMetaData()
        frame_counter = startFrameIndex
        while frame_counter < self.numberOfFrames:
            next_frame = self.getFrameAtIndex(frame_counter)
            next_timestamp = self.getTimestampAtIndex(frame_counter)
            next_frame = _img_type_conversion(
                next_frame,
                self.bitsPerChannel[0],
                self._return_type)
            frame_counter += 1
            if return_ts:
                yield (next_frame, next_timestamp)
            else:
                yield next_frame

    def reader_nextAtIndex(self,
                           frameIndexes,
                           return_ts=False,
                           **kwargs):
        """
        Generator that yields frames and timestamps at given frame indexes.

        Args:
            frameIndexes (iterable): Listlike object containing the indexes of the frames of interest in the video
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False. For compatibility with other video reader classes as timestamps are always None.
            kwargs: implemented for consitency between these methods and those of class VidFileReader.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        self._getMetaData()
        for idx in frameIndexes:
            next_frame = self.getFrameAtIndex(idx)
            next_timestamp = self.getTimestampAtIndex(idx)
            next_frame = _img_type_conversion(
                next_frame,
                self.bitsPerChannel[0],
                self._return_type)
            if return_ts:
                yield (next_frame, next_timestamp)
            else:
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
        self._checkFrameIndex(frameIndex)
        timestamp = self.timestamps[frameIndex]
        return timestamp

    def getAllTimestamps(self):
        try:
            return self.timestamps
        except AttributeError:
            raise AttributeError(
                "Before timestamps can be read, you must read the video meta data by calling the <readMetaData()> method.")

    def _getMetaData(self, framesPerSecond, bitsPerChannel):
        self.frame_names = sorted([fn for fn in os.listdir(
            self.absFileName) if re.search('.'+self.img_type, fn, re.I) != None])
        # timestamps in microsecs estimated based on fixed frame rate and with no offset
        timestamps = (np.arange(
            0, len(self.frame_names)) * 1/framesPerSecond * 1e6)
        self.timestamps = timestamps.astype(np.int64)
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
            "This function is not implemented for this video type.")

    def _getNextTimestamp(self):
        raise NotImplementedError(
            "This function is not implemented for this video type.")

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
