import os

class Config:
    """
    Configuration class for LLMCompress.
    
    This class contains various configuration parameters and utility methods
    used throughout the LLMCompress application.
    """
    
    # Default values for context size and batch size
    CONTEXT_SIZE = 512
    BATCH_SIZE = 64
    
    # Directory paths
    LOGS_DIR = 'Logs'
    RESULTS_DIR = 'Results'
    OUTPUT_DIR = 'Output_Files'
    
    @staticmethod
    def ensure_directories():
        """
        Ensure that all necessary directories exist.
        
        This method creates the logs, results, and output directories
        if they don't already exist.
        """
        for directory in [Config.LOGS_DIR, Config.RESULTS_DIR, Config.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)