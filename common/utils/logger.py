# -*- coding: utf-8 -*-

from loguru import logger

from .file_utils import remove_if_exists


class Logger(object):
    """A logger for print inputs"""

    def __init__(self, rank=0, level='DEBUG', format='{time:YYYY-MM-DD-HH:mm:ss} | {level} | {message}', path=None):
        """Init a logger, remove original log"""
        self.rank = rank
        if path is not None and self.rank == 0:
            remove_if_exists(path)
            logger.add(path, level=level, format=format)

    def add_log(self, msg, level='INFO'):
        """Only allows writing of rank-0 node"""
        if self.rank != 0:
            return

        assert level.lower() in ['info', 'debug', 'warning', 'error'], 'Please input valid log level...'
        if level.lower() == 'info':
            logger.info(msg)
        elif level.lower() == 'debug':
            logger.debug(msg)
        elif level.lower() == 'warning':
            logger.warning(msg)
        elif level.lower() == 'error':
            logger.error(msg)
