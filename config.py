import os

class Config:
    CONTEXT_SIZE = 512
    BATCH_SIZE = 128
    LOGS_DIR = 'Logs'
    RESULTS_DIR = 'Results'
    OUTPUT_DIR = 'Output_Files'
    
    @staticmethod
    def ensure_directories():
        for directory in [Config.LOGS_DIR, Config.RESULTS_DIR, Config.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)