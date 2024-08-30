import os

class Config:
    LOGS_DIR = 'Logs'
    RESULTS_DIR = 'Results'
    OUTPUT_DIR = 'Output_Files'

    # Default values
    CONTEXT_SIZE = 512
    BATCH_SIZE = 64

    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")

    @staticmethod
    def ensure_directories():
        for directory in [Config.LOGS_DIR, Config.RESULTS_DIR, Config.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_output_file_path(cls, input_file, model_name, compression_method):
        base_name = os.path.basename(input_file)
        output_file_name = f"compressed_{model_name}_{compression_method}_{base_name}.gpz"
        return os.path.join(cls.OUTPUT_DIR, output_file_name)