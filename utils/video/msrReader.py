"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 18:41
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Video reader class for a custom video format that is used at the IBMT.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import numpy as np
import struct
from datetime import date, time, datetime


Logger.setGlobalLoglvl('error')


def _convertBinToInt(byteArray, bitsPerChannel, colorModel, byteorder):
    """Convert binary values to integers. Used for the custom MSR video format.
    """
    if byteorder == 'big':
        dataType = np.dtype(np.uint8)
    elif byteorder == 'little':
        dataType = np.dtype(np.uint8).newbyteorder('>')
    if colorModel == 'M':
        if bitsPerChannel[0] == 8:
            # no conversion necessary
            Logger.logWarning(
                'This conversion type (model: {0}, bitdepth: {1:d}) has not been tested yet...'.format(colorModel, bitsPerChannel[0]))
            return np.frombuffer(byteArray, dtype=dataType)
        elif bitsPerChannel[0] == 10:
            Logger.logWarning(
                'This conversion type (model: {0}, bitdepth: {1:d}) has not been tested yet...'.format(colorModel, bitsPerChannel[0]))
            return _binToInt_mono10(np.frombuffer(byteArray, dtype=dataType))
        elif bitsPerChannel[0] == 12:
            Logger.logWarning(
                'This conversion type (model: {0}, bitdepth: {1:d}) has not been tested yet...'.format(colorModel, bitsPerChannel[0]))
            return _binToInt_mono12(np.frombuffer(byteArray, dtype=dataType))
        else:
            Logger.logError(
                'The specified bitsPerChannel have not yet been defined for this colorModel...')
            raise ValueError('bitsPerChannel not defined')

    elif colorModel == 'RGB':
        if bitsPerChannel[0] == 8:
            # no conversion necessary
            Logger.logWarning(
                'This conversion type (model: {0}, bitdepth: {1:d}) has not been tested yet...'.format(colorModel, bitsPerChannel[0]))
            return np.frombuffer(byteArray, dtype=dataType)
        elif bitsPerChannel[0] == 10:
            return _binToInt_rgb10(np.frombuffer(byteArray, dtype=dataType))
        elif bitsPerChannel[0] == 12:
            return _binToInt_rgb12(np.frombuffer(byteArray, dtype=dataType))
        else:
            Logger.logError(
                'The specified bitsPerChannel have not yet been defined for this colorModel {}...'.format(colorModel))
            raise ValueError('bitsPerChannel not defined')

    else:
        Logger.logError(
            'The specified colorModel {} has not yet been defined...'.format(colorModel))
        raise ValueError('ColorModel not defined')


MONO10_SHIFTS = [
    2**2, 1/2**6,
    2**4, 1/2**4,
    2**6, 1/2**2,
    2**8, 1/2**0
]


def _binToInt_mono10(uint8Array):
    bytesPerChunk = 5
    assert np.mod(
        uint8Array.shape[0], bytesPerChunk) == 0, 'modulo operation for uint8Array.shape[0] and bytesPerChunk not zero'

    tmp = uint8Array.copy().reshape((-1, bytesPerChunk))
    bytes_1 = np.sum(tmp[:, 0:2] * MONO10_SHIFTS[0:2],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 1] = tmp[:, 1] % 2**6
    bytes_2 = np.sum(tmp[:, 1:3] * MONO10_SHIFTS[2:4],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 2] = tmp[:, 2] % 2**4
    bytes_3 = np.sum(tmp[:, 2:4] * MONO10_SHIFTS[4:6],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 3] = tmp[:, 3] % 2**2
    bytes_4 = np.sum(tmp[:, 3:] * MONO10_SHIFTS[6:],
                     dtype=np.uint64, axis=1, keepdims=True)

    out = np.concatenate([bytes_1, bytes_2, bytes_3, bytes_4], axis=1)
    # flatten the array back to the input shape
    out = out.reshape((-1,))

    return out


MONO12_SHIFTS = [
    2**4, 1/2**4,
    2**8, 2**0
]


def _binToInt_mono12(uint8Array):
    bytesPerChunk = 3
    # ensure that the uint8Array has the right length
    assert np.mod(
        uint8Array.shape[0], bytesPerChunk) == 0, 'modulo operation for uint8Array.shape[0] and bytesPerChunk not zero'

    tmp = uint8Array.copy().reshape((-1, bytesPerChunk))
    bytes_1 = np.sum(tmp[:, 0:2] * MONO12_SHIFTS[0:2],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 1] = tmp[:, 1] % 2**4
    bytes_2 = np.sum(tmp[:, 1:] * MONO12_SHIFTS[2:],
                     dtype=np.uint64, axis=1, keepdims=True)

    out = np.concatenate([bytes_1, bytes_2], axis=1)
    # flatten the array back to the input shape
    out = out.reshape((-1,))

    return out


RGB10_SHIFTS = [
    2**22, 2**14, 2**6, 1/2**2,
    2**28, 2**20, 2**12, 2**4, 1/2**4,
    2**26, 2**18, 2**10, 2**2, 1/2**6,
    2**24, 2**16, 2**8, 2**0
]


def _binToInt_rgb10(uint8Array):
    bytesPerChunk = 15
    # ensure that the uint8Array has the right length
    assert np.mod(
        uint8Array.shape[0], bytesPerChunk) == 0, 'modulo operation for uint8Array.shape[0] and bytesPerChunk not zero'

    tmp = uint8Array.copy().reshape((-1, bytesPerChunk))
    bytes_1 = np.sum(tmp[:, 0:4] * RGB10_SHIFTS[0:4],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 3] = tmp[:, 3] % 2**2
    bytes_2 = np.sum(tmp[:, 3:8] * RGB10_SHIFTS[4:9],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 7] = tmp[:, 7] % 2**4
    bytes_3 = np.sum(tmp[:, 7:12] * RGB10_SHIFTS[9:14],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 11] = tmp[:, 11] % 2**6
    bytes_4 = np.sum(tmp[:, 11:] * RGB10_SHIFTS[14:],
                     dtype=np.uint64, axis=1, keepdims=True)

    out = np.concatenate([bytes_1, bytes_2, bytes_3, bytes_4], axis=1)
    # flatten the array back to the input shape
    out = out.reshape((-1,))

    return out


RGB12_SHIFTS = [
    2**28, 2**20, 2**12, 2**4, 1/2**4,  # shifts for bytes_1
    2**32, 2**24, 2**16, 2**8, 2**0  # shifts for bytes_2
]


def _binToInt_rgb12(uint8Array):
    bytesPerChunk = 9
    # ensure that the uint8Array has the right length
    assert np.mod(
        uint8Array.shape[0], bytesPerChunk) == 0, 'modulo operation for uint8Array.shape[0] and bytesPerChunk not zero'

    tmp = uint8Array.copy().reshape((-1, bytesPerChunk))
    bytes_1 = np.sum(tmp[:, 0:5] * RGB12_SHIFTS[0:5],
                     dtype=np.uint64, axis=1, keepdims=True)

    tmp[:, 4] = tmp[:, 4] % 2**4
    bytes_2 = np.sum(tmp[:, 4:] * RGB12_SHIFTS[5:],
                     dtype=np.uint64, axis=1, keepdims=True)

    out = np.concatenate([bytes_1, bytes_2], axis=1)
    # flatten the array back to the input shape
    out = out.reshape((-1,))

    return out


class MSRReader():
    """Reads a msr version 1 (*.msr) video file so that it can be used in python. This class is developed based on the MSRReader.m from the matfiles. MSR is a custom video format developed and used internally at the IBMT.
    """

    def __init__(self, absFileName, byteorder='little', img_coord_sys='indices'):
        """Instanciate a reader for msr foramtted videos.

        Arguments:
            absFileName {str} -- absolute path to the video file

        Keyword Arguments:
            byteorder {str} -- little endian or big endian architecture (default: {'little'})
            img_coord_sys {str} -- defines the orientation of the images ('indices' or 'spatial') (default: {'indices'})
        """
        self.absFileName = absFileName
        self.byteorder = byteorder
        # set the byteorder sign for struct.unpack func
        if self.byteorder == 'big':
            self.bos = '>'
        elif self.byteorder == 'little':
            self.bos = '<'
        self.img_coord_sys = img_coord_sys
        self.numberOfFrames = None
        self.resolution = None
        self.bytesPerFrame = None
        self.colorMode = None
        self.fileSize = None
        self.framesPerSecond = None
        self.duration = None
        self.headerSize = 16  # in byte
        self.timestampSize = 8  # in byte
        self.width = None
        self.height = None
        self.colorModel = None
        self.bitsPerChannel = None
        self._frameCounter = None  # only used for reading frame by frame
        self._readMetaData = False  # check if meta data of the video has been read

    def reader_next(self, startFrameIndex=0, return_ts=False):
        """Generator that yields frames and timestamps frame by frame until the video is completly read.

        Args:
            startFrameIndex (int, optional): frame index from which to start reading
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        with open(self.absFileName, 'rb') as f:
            self._getMetaData(f)
            if startFrameIndex != 0:
                offset = self.headerSize + startFrameIndex * \
                    (self.timestampSize + self.bytesPerFrame)
                f.seek(offset, os.SEEK_SET)
            self._frameCounter = startFrameIndex
            while self._frameCounter < self.numberOfFrames:
                next_timestamp = self._getNextTimestamp(f)
                next_frame = self._getNextFrame(f)
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
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        with open(self.absFileName, 'rb') as f:
            self._getMetaData(f)
            if return_ts:
                for idx in frameIndexes:
                    next_timestamp = self.getTimestampAtIndex(f, idx)
                    next_frame = self.getFrameAtIndex(f, idx)
                    yield (next_frame, next_timestamp)
            else:
                for idx in frameIndexes:
                    next_frame = self.getFrameAtIndex(f, idx)
                    yield next_frame

    def readMetaData(self):
        with open(self.absFileName, 'rb') as f:
            self._getMetaData(f)

    def getFrameAtIndex(self, f: 'filehandler', frameIndex):
        """Reading the frame of a specific position defined by frameIndex. The indexing is 0-based.
        """
        if not self._readMetaData:
            self._getMetaData(f)

        self._checkFrameIndex(frameIndex)
        offset = self.headerSize + self.timestampSize + \
            frameIndex * (self.timestampSize + self.bytesPerFrame)
        f.seek(offset, os.SEEK_SET)
        # For 12bit mono (10bit used)
        if self.colorMode == 26:
            Logger.logWarning('This colorMode has not been tested yet...')
            binFrame = f.read(self.bytesPerFrame)
            binFrame = np.frombuffer(binFrame, dtype=np.dtype(
                np.uint16).newbyteorder(self.bos))
            frame = np.reshape((binFrame & 4092)/4,
                               self.resolution, order=self.order)
            return frame
        frame = np.zeros((self.resolution[0], self.resolution[1], 3))
        binFrame = f.read(self.bytesPerFrame)
        binFrame = np.frombuffer(binFrame, dtype=np.dtype(
            np.uint32).newbyteorder(self.bos))

        frame[:, :, 2] = np.reshape(
            (binFrame & 1023), self.resolution, order=self.order)
        frame[:, :, 1] = np.reshape(
            (binFrame & 1047552)/1024, self.resolution, order=self.order)
        frame[:, :, 0] = np.reshape(
            (binFrame & 1072693248)/1048576, self.resolution, order=self.order)
        return frame

    def getTimestampAtIndex(self, f: 'filehandler', frameIndex):
        """Reading the timestamp of a specific position defined by frameIndex. The indexing is 0-based.
        """
        if not self._readMetaData:
            self._getMetaData(f)

        self._checkFrameIndex(frameIndex)
        offset = self.headerSize + (frameIndex) * \
            (self.timestampSize + self.bytesPerFrame)
        f.seek(offset, os.SEEK_SET)
        timestamp = int.from_bytes(
            f.read(8), byteorder=self.byteorder, signed=False)
        if timestamp == int(16*'f', 16):
            if frameIndex == 0 or frameIndex == self.numberOfFrames:
                Logger.logError(
                    'Can\'t interpolate corrupted timestamp at frameIndex '+str(frameIndex))
            # NOTE not sure if this recursion won't brake at some point
            t1 = self.getTimestampAtIndex(f, frameIndex-1)
            t2 = self.getTimestampAtIndex(f, frameIndex+1)
            timestamp = t1+(t2-t1)/2
        return timestamp

    def getAllTimestamps(self, f: 'filehandler'):
        if not self._readMetaData:
            self._getMetaData(f)

        f.seek(self.headerSize, os.SEEK_SET)
        timestamps = []
        for i in range(0, self.numberOfFrames):
            timestamps.append(int.from_bytes(
                f.read(8), byteorder=self.byteorder, signed=False))
            f.seek(self.bytesPerFrame, os.SEEK_CUR)
        timestamps = np.array(timestamps)
        # correct corrupted timestamps (ct) with corrected values (cv)
        ct = np.nonzero(timestamps == int(16*'f', 16))[0]
        # rounding differences comparing matlab and python can occure at the next step (due to round rules)
        cv = np.asarray(
            np.round(np.mean(np.array((timestamps[ct-1], timestamps[ct+1])), axis=0)), 'int')
        timestamps[ct] = cv
        return timestamps

    def _getMetaData(self, f: 'filehandler'):
        Logger.logDebug('Opened file '+self.absFileName)
        self.colorMode = int.from_bytes(
            f.read(4), byteorder=self.byteorder, signed=False)
        self.bytesPerFrame = int.from_bytes(
            f.read(4), byteorder=self.byteorder, signed=False)
        resX = int.from_bytes(
            f.read(4), byteorder=self.byteorder, signed=False)
        resY = int.from_bytes(
            f.read(4), byteorder=self.byteorder, signed=False)
        self._readMetaData = True

        if self.img_coord_sys == 'indices':
            self.resolution = (resY, resX)  # pixel indices
            self.order = 'F'  # fill columns first
        if self.img_coord_sys == 'spatial':
            self.resolution = (resX, resY)  # spatial coordinates
            self.order = 'C'  # fill rows first
        self.fileSize = os.path.getsize(self.absFileName)
        self.numberOfFrames = int(
            (self.fileSize - self.headerSize) / (self.bytesPerFrame + self.timestampSize))
        self.duration = (self.getTimestampAtIndex(
            f, self.numberOfFrames-1) - self.getTimestampAtIndex(f, 0))/10**7  # accuracy equals 100ns
        self.framesPerSecond = self.numberOfFrames / self.duration
        self.width = resX
        self.height = resY
        self.colorModel = 'RGB'
        # type tuple for forward compatibility with msr2 format
        self.bitsPerChannel = (10,)

        self._frameCounter = 0
        self._resetPointer(f)
        [Logger.logDebug('var: ' + key + ', val: ' + str(val))
         for key, val in vars(self).items()]

    def _getNextFrame(self, f: 'filehandler'):
        """Assumes the pointer is directly(!) before the next frame. It is intended
        to be used together with _getNextTimestamp after applying getMetaData.
        """
        if self._frameCounter < self.numberOfFrames:
            # For 12bit mono (10bit used)
            if self.colorMode == 26:
                Logger.logWarning('This colorMode has not been tested yet...')
                binFrame = f.read(self.bytesPerFrame)
                binFrame = np.frombuffer(binFrame, dtype=np.dtype(
                    np.uint16).newbyteorder(self.bos))
                frame = np.reshape((binFrame & 4092)/4,
                                   self.resolution, order=self.order)
                return frame
            frame = np.zeros((self.resolution[0], self.resolution[1], 3))
            binFrame = f.read(self.bytesPerFrame)
            binFrame = np.frombuffer(binFrame, dtype=np.dtype(
                np.uint32).newbyteorder(self.bos))
            frame[:, :, 2] = np.reshape(
                (binFrame & 1023), self.resolution, order=self.order)
            frame[:, :, 1] = np.reshape(
                (binFrame & 1047552)/1024, self.resolution, order=self.order)
            frame[:, :, 0] = np.reshape(
                (binFrame & 1072693248)/1048576, self.resolution, order=self.order)
            return frame
        return None

    def _getNextTimestamp(self, f: 'filehandler'):
        """Assumes the pointer is directly(!) before the next timestamp. It is intended
        to be used together with _getNextFrame.
        """
        if self._frameCounter < self.numberOfFrames:
            timestamp = int.from_bytes(
                f.read(8), byteorder=self.byteorder, signed=False)
            if timestamp == int(16*'f', 16):
                # save current position in file
                lastPos = os.SEEK_CUR
                # NOTE not sure if this recursion won't brake at some point
                t1 = self.getTimestampAtIndex(f, self._frameCounter-1)
                t2 = self.getTimestampAtIndex(f, self._frameCounter+1)
                timestamp = t1+(t2-t1)/2
                # reset pointer to last position
                f.seek(lastPos, os.SEEK_SET)
            return timestamp
        return None

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
        return valid

    def _resetPointer(self, f: 'filehandler'):
        """
        Set the pointer of the filehandler to the position after the metadata.
        """
        f.seek(self.headerSize, os.SEEK_SET)
        self._frameCounter = 0


class MSR2Reader():
    """Reads a msr version 2 (*.msr2) video file so that it can be used in python. This class is developed based on the MSR2ReaderV2.m from the matfiles. MSR2 is a custom video format developed and used internally at the IBMT.
    """

    def __init__(self, absFileName, byteorder='big', img_coord_sys='indices'):
        """Instanciate a reader for msr foramtted videos.

        Arguments:
            absFileName {str} -- absolute path to the video file

        Keyword Arguments:
            byteorder {str} -- little endian or big endian architecture (default: {'little'})
            img_coord_sys {str} -- defines the orientation of the images ('indices' or 'spatial') (default: {'indices'})
        """
        self.absFileName = absFileName
        self.byteorder = byteorder
        # set the byteorder sign for struct.unpack func
        if self.byteorder == 'big':
            self.bos = '>'
        elif self.byteorder == 'little':
            self.bos = '<'
        self.img_coord_sys = img_coord_sys
        self.numberOfFrames = None
        self.resolution = None
        self.bytesPerFrame = None
        self.fileSize = None
        self.framesPerSecond = None
        self.duration = None
        self.headerSize = None
        self.timestampSize = 8  # in byte
        self.width = None
        self.height = None
        self.colorModel = None
        self.bitsPerChannel = None
        self.cameraModel = None
        self.firstCompStamp = None
        self.timerFreq = 10**7

        self._rgb12 = False
        self._mono12 = False
        self._rgb10 = False
        self._rgb8 = False
        self._mono8 = False

        self._deadBitsPerFrame = 0
        self._accE = 3  # Accepted error in frame rate (in percent)

        self._frameCounter = None  # only used for reading frame by frame
        self._readMetaData = False  # check if meta data of the video has been read

    def reader_next(self, startFrameIndex=0, return_ts=False):
        """Generator that yields frames and timestamps frame by frame until the video is completly read.

        Args:
            startFrameIndex (int, optional): frame index from which to start reading
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        with open(self.absFileName, 'rb') as f:
            self._getMetaData(f)
            if startFrameIndex != 0:
                offset = self.headerSize + startFrameIndex * \
                    (self.timestampSize + self.bytesPerFrame)
                f.seek(offset, os.SEEK_SET)
            self._frameCounter = startFrameIndex
            while self._frameCounter < self.numberOfFrames:
                next_timestamp = self._getNextTimestamp(f)
                next_frame = self._getNextFrame(f)
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
            return_ts (bool, optional): Set if the timestamps from the camera should also be returned. Defaults to False.

        Yields:
            tuple or array: tuple for return_ts=True and numpy array otherwise.
        """
        with open(self.absFileName, 'rb') as f:
            self._getMetaData(f)
            if return_ts:
                for idx in frameIndexes:
                    next_timestamp = self.getTimestampAtIndex(f, idx)
                    next_frame = self.getFrameAtIndex(f, idx)
                    yield (next_frame, next_timestamp)
            else:
                for idx in frameIndexes:
                    next_frame = self.getFrameAtIndex(f, idx)
                    yield next_frame

    def readMetaData(self):
        with open(self.absFileName, 'rb') as f:
            self._getMetaData(f)

    def getFrameAtIndex(self, f: 'filehandler', frameIndex):
        """Reading the frame of a specific position defined by frameIndex. The indexing is 0-based.
        """
        if not self._readMetaData:
            self._getMetaData(f)

        self._checkFrameIndex(frameIndex)
        offset = self.headerSize + self.timestampSize + \
            frameIndex * (self.timestampSize + self.bytesPerFrame)
        f.seek(offset, os.SEEK_SET)

        binFrame = f.read(self.bytesPerFrame)
        binFrame = _convertBinToInt(
            binFrame, self.bitsPerChannel, self.colorModel, self.byteorder)

        # For Mono colorMode
        if 'M' in self.colorModel:
            Logger.logWarning('This colorMode has not been tested yet...')
            frame = np.reshape(binFrame, self.resolution, order=self.order)
            return frame

        if 'RGB' in self.colorModel:
            bpc = self.bitsPerChannel[0]
            frame = np.zeros((self.resolution[0], self.resolution[1], 3))
            frame[:, :, 2] = np.reshape(
                (binFrame & (2**bpc-1)), self.resolution, order=self.order)
            frame[:, :, 1] = np.reshape(
                (binFrame & (2**(bpc*2)-2**bpc))/(2**bpc), self.resolution, order=self.order)
            frame[:, :, 0] = np.reshape(
                (binFrame & (2**(bpc*3)-2**(bpc*2)))/(2**(bpc*2)), self.resolution, order=self.order)
            return frame

    def getTimestampAtIndex(self, f: 'filehandler', frameIndex):
        """Reading the timestamp of a specific position defined by frameIndex. The indexing is 0-based.
        """
        if not self._readMetaData:
            self._getMetaData(f)

        self._checkFrameIndex(frameIndex)
        offset = self.headerSize + (frameIndex) * \
            (self.timestampSize + self.bytesPerFrame)
        f.seek(offset, os.SEEK_SET)
        timestamp = int.from_bytes(
            f.read(8), byteorder=self.byteorder, signed=False)
        if timestamp == int(16*'f', 16):
            if frameIndex == 0 or frameIndex == self.numberOfFrames:
                Logger.logError(
                    'Can\'t interpolate corrupted timestamp at frameIndex '+str(frameIndex))
            # NOTE not sure if this recursion won't brake at some point
            t1 = self.getTimestampAtIndex(f, frameIndex-1)
            t2 = self.getTimestampAtIndex(f, frameIndex+1)
            timestamp = t1+(t2-t1)/2
        return timestamp

    def getAllTimestamps(self, f: 'filehandler'):
        if not self._readMetaData:
            self._getMetaData(f)

        f.seek(self.headerSize, os.SEEK_SET)
        timestamps = []
        for i in range(0, self.numberOfFrames):
            timestamps.append(int.from_bytes(
                f.read(8), byteorder=self.byteorder, signed=False))
            f.seek(self.bytesPerFrame, os.SEEK_CUR)
        timestamps = np.array(timestamps)
        # correct corrupted timestamps (ct) with corrected values (cv)
        ct = np.nonzero(timestamps == int(16*'f', 16))[0]
        # rounding differences comparing matlab and python can occure at the next step (due to round rules)
        cv = np.asarray(
            np.round(np.mean(np.array((timestamps[ct-1], timestamps[ct+1])), axis=0)), 'int')
        timestamps[ct] = cv
        return timestamps

    def _getMetaData(self, f: 'filehandler'):
        Logger.logDebug('Opened file '+self.absFileName)
        self.headerSize = int(int.from_bytes(
            f.read(8), byteorder=self.byteorder, signed=False)/8)  # /8 because header size is in bit
        self.colorModel = ((f.read(8)).decode('windows-1252')).split('\x00')[0]

        if ('M' not in self.colorModel) and ('RGB' not in self.colorModel):
            Logger.logError(
                'The color mode is not supported: colorModel' + self.colorModel)
            raise ValueError('The color model - ' +
                             self.colorModel + ' - is not supported')

        self.bitsPerChannel = np.array(struct.unpack(
            8*'B', f.read(8))[0:len(self.colorModel)])

        self.bytesPerFrame = None

        self.width = int.from_bytes(
            f.read(4), byteorder=self.byteorder, signed=False)
        self.height = int.from_bytes(
            f.read(4), byteorder=self.byteorder, signed=False)
        self.framesPerSecond = struct.unpack(self.bos+'d', f.read(8))[0]
        self.numberOfFrames = int.from_bytes(
            f.read(8), byteorder=self.byteorder, signed=False)

        f.seek(8, os.SEEK_CUR)  # skip timer freq for now (??)
        self.cameraModel = (f.read(32)).split(b'\x00')[0].decode('utf-8')

        if self.headerSize > 88:
            tmp_time = int.from_bytes(
                f.read(8), byteorder=self.byteorder, signed=False)
            if tmp_time > 10**10:
                tmp_time = tmp_time/86400000 + date.toordinal(date(1970, 1, 1))
            else:
                tmp_time = tmp_time/86400 + date.toordinal(date(1970, 1, 1))
            d = date.fromordinal(int(tmp_time))
            hrs = (tmp_time % 1)*24
            mins = (hrs % 1)*60
            secs = (mins % 1)*60
            msecs = (secs % 1)*1000
            t = time(hour=int(hrs), minute=int(mins),
                     second=int(secs), microsecond=int(msecs))
            self.firstCompStamp = datetime.combine(d, t)

        if self.colorModel == 'RGB' and sum(self.bitsPerChannel-12) == 0:
            s = 3
            self._rgb12 = True

        elif self.colorModel == 'RGB' and sum(self.bitsPerChannel-10) == 0:
            s = 3
            self._rgb10 = True

        elif self.colorModel == 'RGB' and sum(self.bitsPerChannel-8) == 0:
            s = 3
            self._rgb8 = True

        elif self.colorModel == 'M' and self.bitsPerChannel[0] == 12:
            s = 1
            self._mono12 = True

        elif self.colorModel == 'M' and self.bitsPerChannel[0] == 8:
            s = 1
            self._mono8 = True

        # for RGB, Mono with symmetric BitDepths
        bitCount = self.width * self.height * self.bitsPerChannel[0] * s
        self._deadBitsPerFrame = -bitCount % 8
        if self._deadBitsPerFrame != 0:
            Logger.logError(
                'This version does not support videos with dead bits at the end of the frames yet')
            raise NotImplementedError('Dead bits per frame not supported')

        self.bytesPerFrame = int((bitCount + self._deadBitsPerFrame)/8)
        self._readMetaData = True

        if self.img_coord_sys == 'indices':
            self.resolution = (self.width, self.height)  # pixel indices
            self.order = 'F'  # fill columns first
        if self.img_coord_sys == 'spatial':
            self.resolution = (self.height, self.width)  # spatial coordinates
            self.order = 'C'  # fill rows first
        self.fileSize = os.path.getsize(self.absFileName)
        self.duration = (self.getTimestampAtIndex(
            f, self.numberOfFrames-1) - self.getTimestampAtIndex(f, 0))/self.timerFreq

        self._frameCounter = 0
        self._resetPointer(f)
        [Logger.logDebug('var: ' + key + ', val: ' + str(val))
         for key, val in vars(self).items()]

    def _getNextFrame(self, f: 'filehandler'):
        """Assumes the pointer is directly(!) before the next frame. It is intended
        to be used together with _getNextTimestamp after applying getMetaData.
        """
        if self._frameCounter < self.numberOfFrames:
            binFrame = f.read(self.bytesPerFrame)
            binFrame = _convertBinToInt(
                binFrame, self.bitsPerChannel, self.colorModel, self.byteorder)
            # For Mono colorMode
            if 'M' in self.colorModel:
                Logger.logWarning('This colorMode has not been tested yet...')
                frame = np.reshape(binFrame, self.resolution, order=self.order)
                return frame

            if 'RGB' in self.colorModel:
                bpc = self.bitsPerChannel[0]
                frame = np.zeros((self.resolution[0], self.resolution[1], 3))
                frame[:, :, 2] = np.reshape(
                    (binFrame & (2**bpc-1)), self.resolution, order=self.order)
                frame[:, :, 1] = np.reshape(
                    (binFrame & (2**(bpc*2)-2**bpc))/(2**bpc), self.resolution, order=self.order)
                frame[:, :, 0] = np.reshape(
                    (binFrame & (2**(bpc*3)-2**(bpc*2)))/(2**(bpc*2)), self.resolution, order=self.order)
                return frame
        return None

    def _getNextTimestamp(self, f: 'filehandler'):
        """Assumes the pointer is directly(!) before the next timestamp. It is intended
        to be used together with _getNextFrame.
        """
        if self._frameCounter < self.numberOfFrames:
            timestamp = int.from_bytes(
                f.read(8), byteorder=self.byteorder, signed=False)
            if timestamp == int(16*'f', 16):
                # save current position in file
                lastPos = os.SEEK_CUR
                # NOTE not sure if this recursion won't brake at some point
                t1 = self.getTimestampAtIndex(f, self._frameCounter-1)
                t2 = self.getTimestampAtIndex(f, self._frameCounter+1)
                timestamp = t1+(t2-t1)/2
                # reset pointer to last position
                f.seek(lastPos, os.SEEK_SET)
            return timestamp
        return None

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

    def _resetPointer(self, f: 'filehandler'):
        """
        Set the pointer of the filehandler to the position after the metadata.
        """
        f.seek(self.headerSize, os.SEEK_SET)
        self._frameCounter = 0

    def _getFrameRateErrors(self):
        ds_des = self.timerFreq/self.framesPerSecond
        stamps = self.getAllTimestamps()
        ds = stamps[1:] - stamps[0:-1]

        mask = np.logical_or((ds > (1+self._accE/100) * ds_des),
                             (ds < (1-self._accE/100)*ds_des))
        mask = np.concatenate([mask, np.array([False])])
        ind = np.nonzero(mask)
        return ind, mask, stamps
