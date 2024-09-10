import logging
from tqdm import tqdm
from config import Config
import os

class TqdmToLogger:
    """
    A class to redirect tqdm output to a logger.
    """
    def __init__(self, logger, level=logging.INFO, interval=5):
        """
        Initialize the TqdmToLogger.

        Args:
            logger (logging.Logger): The logger object to use.
            level (int): The logging level (default: logging.INFO).
            interval (int): The interval at which to log messages (default: 5).
        """
        self.logger = logger
        self.level = level
        self.interval = interval

    def __call__(self, msg):
        """
        Log a message at specified intervals.

        Args:
            msg (str): The message to log.
        """
        if self.interval and hasattr(self, "counter"):
            self.counter += 1
            if self.counter % self.interval == 0:
                self.logger.log(self.level, msg)
        else:
            self.counter = 1
            self.logger.log(self.level, msg)

class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that writes messages using tqdm.write().
    """
    def emit(self, record):
        """
        Emit a log record.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging():
    """
    Set up logging for the LLMCompress application.
    
    This function configures logging to write to both a file and the console,
    using custom handlers for tqdm compatibility.
    """
    Config.ensure_directories()
    file_handler = logging.FileHandler(os.path.join(Config.LOGS_DIR, 'LLMCompress.log'), mode='w')
    console_handler = TqdmLoggingHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])