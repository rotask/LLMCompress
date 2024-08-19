import logging
from tqdm import tqdm
from config import Config
import os

class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO, interval=5):
        self.logger = logger
        self.level = level
        self.interval = interval

    def __call__(self, msg):
        if self.interval and hasattr(self, "counter"):
            self.counter += 1
            if self.counter % self.interval == 0:
                self.logger.log(self.level, msg)
        else:
            self.counter = 1
            self.logger.log(self.level, msg)

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging():
    Config.ensure_directories()
    file_handler = logging.FileHandler(os.path.join(Config.LOGS_DIR, 'LLMZip.log'), mode='w')
    console_handler = TqdmLoggingHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])