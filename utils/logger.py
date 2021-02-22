"""
-------------------------------------------------------------------------------
Created: 22.02.2021, 09:20
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Custom logger class for logging events.
-------------------------------------------------------------------------------
"""
import logging
import datetime
import os
import inspect


class Logger():
    """A logger for logging events to a file or the the console.
    """

    ABSOLUTEFILEPATH = None
    INSTANCES = {}
    LOGLVL_SH = logging.DEBUG
    LOGLVL_FH = logging.DEBUG

    def __init__(self, caller_module_name):
        logger = logging.getLogger(caller_module_name)
        logger.setLevel(logging.DEBUG)
        fh = None
        if Logger.ABSOLUTEFILEPATH is not None:
            file_name = '{}.log'.format(caller_module_name)
            fh = logging.FileHandler(os.path.join(
                Logger.ABSOLUTEFILEPATH, file_name))
            fh.setFormatter(CustomFormatter())
            fh.setLevel(Logger.LOGLVL_FH)
        sh = logging.StreamHandler()
        sh.setFormatter(CustomFormatter())
        sh.setLevel(Logger.LOGLVL_SH)
        logger.addHandler(sh)
        if fh is not None:
            logger.addHandler(fh)
        Logger.INSTANCES[caller_module_name] = logger

    @staticmethod
    def setGlobalLoglvl(lvl: str):
        if lvl == 'debug':
            Logger.LOGLVL_SH = logging.DEBUG
            Logger.LOGLVL_FH = logging.DEBUG
        elif lvl == 'info':
            Logger.LOGLVL_SH = logging.INFO
            Logger.LOGLVL_FH = logging.INFO
        elif lvl == 'warning':
            Logger.LOGLVL_SH = logging.WARNING
            Logger.LOGLVL_FH = logging.WARNING
        elif lvl == 'critical':
            Logger.LOGLVL_SH = logging.CRITICAL
            Logger.LOGLVL_FH = logging.CRITICAL
        elif lvl == 'error':
            Logger.LOGLVL_SH = logging.ERROR
            Logger.LOGLVL_FH = logging.ERROR
        else:
            raise NotImplementedError(
                'This loglvl <<{}>> is not implemented!'.format(lvl))

    @staticmethod
    def setAbsFilePath(path: str, set_global: bool = False):
        if set_global:
            instance = Logger
        else:
            instance = Logger._getInstance()
        instance.ABSOLUTEFILEPATH = path

    @staticmethod
    def logDebug(msg):
        instance = Logger._getInstance()
        instance.debug(msg)

    @staticmethod
    def logInfo(msg):
        instance = Logger._getInstance()
        instance.info(msg)

    @staticmethod
    def logWarning(msg):
        instance = Logger._getInstance()
        instance.warning(msg)

    @staticmethod
    def logError(msg):
        instance = Logger._getInstance()
        instance.error(msg)

    @staticmethod
    def logCritical(msg):
        instance = Logger._getInstance()
        instance.critical(msg)

    @staticmethod
    def _getInstance():
        frame_records = inspect.stack()[2]
        caller_module_name = inspect.getmodulename(frame_records[1])
        if caller_module_name not in Logger.INSTANCES.keys():
            Logger(caller_module_name)
        return Logger.INSTANCES[caller_module_name]


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    Bold = "\033[1m"
    Dim = "\033[2m"
    Underlined = "\033[4m"
    Blink = "\033[5m"
    Reverse = "\033[7m"
    Hidden = "\033[8m"

    ResetBold = "\033[21m"
    ResetDim = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink = "\033[25m"
    ResetReverse = "\033[27m"
    ResetHidden = "\033[28m"

    ResetAll = "\033[0m"

    Default = "\033[39m"
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"

    BASE_FORMAT = '%(asctime)s - %(name)-16s - %(levelname)-10s - %(message)s (%(filename)s:%(lineno)d)'

    FORMATS = {
        logging.DEBUG: White + BASE_FORMAT + ResetAll,
        logging.INFO: Green + BASE_FORMAT + ResetAll,
        logging.WARNING: Yellow + BASE_FORMAT + ResetAll,
        logging.ERROR: Red + BASE_FORMAT + ResetAll,
        logging.CRITICAL: Bold + Red + BASE_FORMAT + ResetAll,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


if __name__ == '__main__':
    Logger.logInfo('This is a test.')
