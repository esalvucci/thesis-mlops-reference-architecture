from enum import Enum
import logging
from logging.handlers import RotatingFileHandler

component_name = "model_loader"
log_format = '[%(asctime)s] - %(message)s'


class SingletonLogger(Enum):
    @staticmethod
    def get_logger():
        # logger configuration
        logger = logging.getLogger(component_name)
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(component_name + '.log',
                                      maxBytes=100000,
                                      backupCount=5)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger
