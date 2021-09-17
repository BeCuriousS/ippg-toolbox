"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 20:04
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Signal reader class for reading files in the unisens format which is used at the IBMT.
-------------------------------------------------------------------------------
"""
from ..logger import Logger
import os
import numpy as np
import struct
import xml.etree.ElementTree as ET
import csv


Logger.setGlobalLoglvl('error')


class UnisensSigReader():
    """
    Reads a unisens binaray (*.bin) and csv (*.csv) files according to chosen properties. This class
    focuses on the documentation from https://github.com/Unisens/unisens4java and specifically from 
    https://github.com/Unisens/unisens4java/blob/master/documentation/data-example/data-example.pdf
    """

    TREENAME = 'unisens.xml'

    def __init__(self, full_file_name, is_time_file=False):
        """
        Initialize the unisens reader for a chosen file. Assumes that the unisens.xml file lies in
        the same directory.

        Arguments:
            full_file_name {str} -- Absolute file path to the data file. Assumes that there is
            a unisens.xml file in the same directory describing the binary data.
            is_time_file {bool} -- time files must be treated separately because of some weird format specifications.

        """
        self.full_file_name = full_file_name
        self.unisensDir, self.completeFileName = os.path.split(
            self.full_file_name)
        self.is_time_file = is_time_file
        self._readXMLTree()

    def readAllValues(self):
        """
        Returns an array where the columns (1-dim) represent the channels and the 
        rows (0-dim) represent one sample.
        """
        ###########################################################
        # differentiate the file types
        ###########################################################
        if self.completeFileName.endswith('.bin'):
            return self._readAllValuesFromBin()
        elif self.completeFileName.endswith('.csv'):
            return self._readAllValuesFromCSV()
        else:
            raise NotImplementedError(
                'The file type can not (yet) be read by this implementation')

    def readMetaData(self):
        ###########################################################
        # differentiate the file types
        ###########################################################
        if self.completeFileName.endswith('.bin'):
            return self._getMetaDataFromBin()
        elif self.completeFileName.endswith('.csv'):
            return self._getMetaDataFromCSV()
        else:
            raise NotImplementedError(
                'The file type can not (yet) be read by this implementation')

    def _readAllValuesFromBin(self):
        self._getMetaDataFromBin()
        with open(self.full_file_name, 'rb') as f:
            buffer = f.read()
        values = np.frombuffer(buffer, dtype=np.dtype(
            self.dataType).newbyteorder(self.endianInfo))
        if self.is_time_file:
            return np.reshape(values, (-1, 2))[:, 0]
        return np.reshape(values, (-1, self.numberOfChannels))

    def _readAllValuesFromCSV(self):
        """
        This method is implemented for the camera timestamp aquisition from csv files. Only tested with those so far.
        """
        self._getMetaDataFromCSV()
        values = np.loadtxt(self.full_file_name, dtype=float,
                            delimiter=self.delimiter)
        return values[:, 0].astype(np.int)

    def _readXMLTree(self):
        tree = ET.parse(os.path.join(
            self.unisensDir, self.TREENAME))
        root = tree.getroot()
        self.signals = []
        for child in root:
            if 'signalEntry' in child.tag or 'valuesEntry' in child.tag:
                self.signals.append(child)

    def _getMetaDataFromBin(self):
        """
        Extract the data from the xml tree.
        """
        # search for the signal specification in the xml tree
        signal = None
        for sig in self.signals:
            if sig.attrib['id'] == self.completeFileName:
                signal = sig
        if signal is None:
            Logger.logError(
                'The chosen file does not exist in the unisensDir...')
            raise FileExistsError(
                'The chosen file does not exist in the specified path')
            return
        # get the relevant information from the xml tree
        self._getDataType(signal)
        self._getNumberOfChannels(signal)
        self._getEndianInfo(signal)
        self._getLsbValue(signal)
        self._getSampleRate(signal)

        [Logger.logDebug('var: ' + key + ', val: ' + str(val))
         for key, val in vars(self).items()]

    def _getMetaDataFromCSV(self):
        """
        Extract the data from the xml tree.
        """
        # search for the signal specification in the xml tree
        signal = None
        for sig in self.signals:
            if sig.attrib['id'] == self.completeFileName:
                signal = sig
        if signal is None:
            Logger.logError(
                'The chosen file does not exist in the unisensDir...')
            raise FileExistsError(
                'The chosen file does not exist in the specified path')
            return
        # get the relevant information from the xml tree
        self._getDataType(signal)
        self._getNumberOfChannels(signal)
        self._getCSVFileFormat(signal)
        self._getSampleRate(signal)

        [Logger.logDebug('var: ' + key + ', val: ' + str(val))
         for key, val in vars(self).items()]

    def _getCSVFileFormat(self, signalXMLTree):
        self.decimalSeparator = None
        self.delimiter = None
        for child in signalXMLTree:
            if 'csvFileFormat' in child.tag:
                self.decimalSeparator = child.attrib['decimalSeparator']
                self.delimiter = child.attrib['separator']
        if self.decimalSeparator is None or self.delimiter is None:
            Logger.logError(
                'CSV file format specifications not found...')
            raise ValueError('CSV file format specifications not found...')

    def _getDataType(self, signalXMLTree):
        if signalXMLTree.attrib['dataType'] == 'double':
            self.dataType = np.float64
        else:
            Logger.logError('The datatype has not yet been defined...')
            raise NotImplementedError(
                'The datatype of this signal is not implemented yet')
        if self.is_time_file:
            self.dataType = np.int64

    def _getLsbValue(self, signalXMLTree):
        if signalXMLTree.attrib['lsbValue'] == '1':
            self.lsbValue = 1
            Logger.logDebug(
                'Note that the bitorder is not controlled here within...')
        else:
            Logger.logError(
                'The bit orientation in the binary file is not yet implemented...')
            raise NotImplementedError(
                'The bit orientation of this signal is not implemented yet')

    def _getSampleRate(self, signalXMLTree):
        self.sampleRate = int(signalXMLTree.attrib['sampleRate'])

    def _getEndianInfo(self, signalXMLTree):
        self.endianInfo = None
        for child in signalXMLTree:
            if 'binFileFormat' in child.tag:
                if child.attrib['endianess'] == 'LITTLE':
                    self.endianInfo = '<'  # defined by the struct module
                elif child.attrib['endianess'] == 'BIG':
                    self.endianInfo = '>'  # defined by the struct module
        if self.endianInfo is None:
            Logger.logError(
                'There is no endianess information specified for the binary file...')
            raise NotImplementedError(
                'There is no endianess information specified')

    def _getNumberOfChannels(self, signalXMLTree):
        self.numberOfChannels = 0
        for child in signalXMLTree:
            if 'channel' in child.tag:
                self.numberOfChannels += 1
        if self.numberOfChannels == 0:
            Logger.logError('There is no channel set in the chosen file...')
            raise NotImplementedError(
                'Channel number not specified in the file')
