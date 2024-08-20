from llmzip import LLMZip
from logger import setup_logging
from config import Config
import logging
import time
import traceback
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

def main():
    setup_logging()
    Config.ensure_directories()
    
# Mixtral - out of memory for 1 L4
# gpt2 - works with Ranks
# Yi - out of memory for 1 L4
# Nemo - Mistral-Nemo-Base-2407  - works with Ranks
# llama_2 - out of memory for 1 L4
# llama_3 - works with Ranks
# llama_3.1 - works with Ranks
# t5-small - complex debug

    model_name = "llama_3.1"  # Change this to test different models
    compression_method = "AC"  # or "AC or Ranks"
    
    llmzip = LLMZip(model_name, compression_method)
    
    input_file = "Data/small_txt.txt"  # Change this to test different datasets
    output_file = os.path.join(Config.OUTPUT_DIR, f"compressed_{os.path.basename(input_file)}.gpz")
    
    try:
        start_time = time.time()
        llmzip.zip(input_file, output_file)
        end_time = time.time()
        
        logging.info(f"Compression completed in {end_time - start_time:.2f} seconds")
        
        # Optionally, test decompression and check if files match
        decompressed_file = os.path.join(Config.OUTPUT_DIR, f"decompressed_{os.path.basename(input_file)}")
        llmzip.unzip(output_file, decompressed_file)
        llmzip.check(input_file, decompressed_file)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        error_trace = traceback.format_exc()
        logging.error(error_trace)       

if __name__ == "__main__":
    main()