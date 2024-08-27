from llmzip import LLMZip
from logger import setup_logging
from config import Config
import logging
import time
import traceback
import os
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    setup_logging()
    Config.ensure_directories()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    summary = "Summary: Anarchism is a political philosophy that advocates for the abolition of rulers and authoritarian institutions, promoting a society based on voluntary association, mutual aid, and self-governance. The term originated as an insult during the English and French revolutions but was later adopted positively by self-defined anarchists. Key figures in anarchist thought include William Godwin, Pierre-Joseph Proudhon, and Peter Kropotkin. Anarchism encompasses various movements, including anarcho-communism, individualist anarchism, and anarcho-syndicalism, each with differing views on economics and social organization."

    # Mixtral - out of memory for 1 L4
    # gpt2 - works with Ranks
    # Yi - out of memory for 1 L4
    # Nemo - Mistral-Nemo-Base-2407  - works with Ranks
    # llama_2 - out of memory for 1 L4
    # llama_3 - works with Ranks
    # llama_3.1 - works with Ranks
    # Mistral_7B - complex debug
    
    model_name = "gpt2"  # Change this to test different models
    compression_method = "Ranks"  # or "AC or Ranks"
    
    llmzip = LLMZip(model_name, compression_method)

    # List of input files to process
    input_files = [
        # "Data/bookcorpus_1MB.txt",
        # "Data/bookcorpus_1MB_with_summary.txt",
        "Data/text8_1MB.txt",
        # "Data/text8_1MB_with_summary.txt"
    ]

    # List of CONTEXT_SIZE values to iterate over
    context_sizes = [256]

    for context_size in context_sizes:
        Config.CONTEXT_SIZE = context_size  # Update CONTEXT_SIZE in Config

        for input_file in input_files:
            output_file = os.path.join(Config.OUTPUT_DIR, f"compressed_{context_size}_{os.path.basename(input_file)}.gpz")
            
            try:
                # Log CONTEXT_SIZE and BATCH_SIZE from Config
                logging.info(f"CONTEXT_SIZE: {Config.CONTEXT_SIZE}")
                logging.info(f"BATCH_SIZE: {Config.BATCH_SIZE}")
                
                start_time = time.time()
                llmzip.zip(input_file, output_file, summary, gpu_name)
                end_time = time.time()
                
                logging.info(f"Compression of {input_file} with CONTEXT_SIZE={context_size} completed in {end_time - start_time:.2f} seconds")
                
                # Optionally, test decompression and check if files match
                # decompressed_file = os.path.join(Config.OUTPUT_DIR, f"decompressed_{context_size}_{os.path.basename(input_file)}")
                # llmzip.unzip(output_file, decompressed_file)
                # llmzip.check(input_file, decompressed_file)
                
            except Exception as e:
                logging.error(f"An error occurred while processing {input_file} with CONTEXT_SIZE={context_size}: {e}")
                error_trace = traceback.format_exc()
                logging.error(error_trace)

if __name__ == "__main__":
    main()
