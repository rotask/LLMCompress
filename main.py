import argparse
from llmcompress import LLMCompress
from logger import setup_logging
from config import Config
import logging
import time
import traceback
import os

# Set environment variable for PyTorch memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

def get_available_models():
    """
    Returns a list of available models for compression.
    
    Returns:
        list: A list of supported model names.
    """
    return [
        "Mixtral", "gpt2", "Yi", "Nemo", "llama_2", 
        "llama_3", "llama_3.1", "Mistral_7B"
    ]

def get_compression_methods():
    """
    Returns a list of available compression methods.
    
    Returns:
        list: A list of supported compression method names.
    """
    return ["Ranks", "AC"]

def parse_arguments():
    """
    Parses command-line arguments for the LLMCompress tool.
    
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run LLMCompress experiments")
    parser.add_argument("--model", choices=get_available_models(), default="gpt2",
                        help="Name of the model to use")
    parser.add_argument("--compression_method", choices=get_compression_methods(), default="Ranks",
                        help="Compression method to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing (default: 32)")
    parser.add_argument("--context_size", type=int, default=512,
                        help="Context size for processing (default: 512)")
    parser.add_argument("--input_file", type=str, default="Data/text8_1MB.txt",
                        help="Path to the input file")
    return parser.parse_args()

def main():
    """
    Main function to run the LLMCompress compression and decompression process.
    
    This function sets up logging, parses command-line arguments, initializes the LLMCompress object,
    and runs the compression and decompression processes. It also handles exceptions and logs errors.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging()
    
    # Ensure necessary directories exist
    Config.ensure_directories()
    
    # Update Config with command-line arguments
    Config.BATCH_SIZE = args.batch_size
    Config.CONTEXT_SIZE = args.context_size

    # Initialize LLMCompress object with specified model and compression method
    llmzip = LLMCompress(args.model, args.compression_method)
    
    # Define output file path
    output_file = os.path.join(Config.OUTPUT_DIR, f"compressed_{os.path.basename(args.input_file)}.gpz")
    
    try:
        # Compress the input file
        start_time = time.time()
        llmzip.zip(args.input_file, output_file)
        end_time = time.time()
        
        logging.info(f"Compression completed in {end_time - start_time:.2f} seconds")
        
        # Decompress the file and check if it matches the original
        decompressed_file = os.path.join(Config.OUTPUT_DIR, f"decompressed_{os.path.basename(args.input_file)}")
        llmzip.unzip(output_file, decompressed_file)
        llmzip.check(args.input_file, decompressed_file)
        
    except Exception as e:
        # Log any errors that occur during the process
        logging.error(f"An error occurred: {e}")
        error_trace = traceback.format_exc()
        logging.error(error_trace)       

if __name__ == "__main__":
    main()