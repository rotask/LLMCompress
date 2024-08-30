import argparse
from llmzip import LLMZip
from logger import setup_logging
from config import Config
import logging
import time
import traceback
import os
import torch

def get_available_models():
    return [
        "Mixtral", "gpt2", "Yi", "Nemo", "llama_2", 
        "llama_3", "llama_3.1", "Mistral_7B"
    ]

def get_compression_methods():
    return ["Ranks", "AC"]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLMZip experiments")
    parser.add_argument("--model", choices=get_available_models(), default="gpt2",
                        help="Name of the model to use")
    parser.add_argument("--compression_method", choices=get_compression_methods(), default="Ranks",
                        help="Compression method to use")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for processing (default: 64)")
    parser.add_argument("--context_size", type=int, default=512,
                        help="Context size for processing (default: 512)")
    parser.add_argument("--input_file", type=str, default="Data/text8_1MB.txt",
                        help="Path to the input file")
    parser.add_argument("--summary", type=str, default="",
                        help="Summary text to use as side information")
    
    # Parse known args, ignoring any extra args
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = parse_arguments()
    setup_logging()
    Config.ensure_directories()
    
    # Update Config with command-line arguments or defaults
    Config.update(
        BATCH_SIZE=args.batch_size,
        CONTEXT_SIZE=args.context_size
    )

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    llmzip = LLMZip(args.model, args.compression_method)

    try:
        logging.info(f"Starting experiment with the following parameters:")
        logging.info(f"Model: {args.model}")
        logging.info(f"Compression Method: {args.compression_method}")
        logging.info(f"Batch Size: {Config.BATCH_SIZE}")
        logging.info(f"Context Size: {Config.CONTEXT_SIZE}")
        logging.info(f"Input File: {args.input_file}")

        output_file = Config.get_output_file_path(args.input_file, args.model, args.compression_method)
        logging.info(f"Output File: {output_file}")

        start_time = time.time()
        llmzip.zip(args.input_file, output_file, args.summary, gpu_name)
        end_time = time.time()

        logging.info(f"Compression completed in {end_time - start_time:.2f} seconds")

        # Optionally, test decompression and check if files match
        decompressed_file = os.path.join(Config.OUTPUT_DIR, f"decompressed_{os.path.basename(args.input_file)}")
        llmzip.unzip(output_file, decompressed_file)
        llmzip.check(args.input_file, decompressed_file)

    except Exception as e:
        logging.error(f"An error occurred during the experiment: {e}")
        error_trace = traceback.format_exc()
        logging.error(error_trace)

if __name__ == "__main__":
    main()