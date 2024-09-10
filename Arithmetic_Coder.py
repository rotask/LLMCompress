import torch
import numpy as np
import os
import csv
import sys
import logging
from tqdm import tqdm
import constriction
from datetime import datetime
import time
from config import Config

def save_to_csv(model_name, language, num_characters, num_tokens, ratio, entropy, compression_ratio, redundancy, time_taken):
    """
    Save compression results to a CSV file.

    Args:
        model_name (str): Name of the model used for compression.
        language (str): Language of the compressed text.
        num_characters (int): Number of characters in the original text.
        num_tokens (int): Number of tokens in the tokenized text.
        ratio (float): Ratio of characters to tokens.
        entropy (float): Calculated entropy of the compressed data.
        compression_ratio (float): Achieved compression ratio.
        redundancy (float): Calculated redundancy.
        time_taken (float): Time taken for compression in minutes.
    """
    csv_filename = os.path.join(Config.RESULTS_DIR, f'{model_name}AC_compression_results.csv')
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='') as csv_file:
        fieldnames = ['Model', 'Language', 'Characters', 'Tokens', 'Ratio', 'Entropy', 'Compression Ratio', 'Redundancy', 'Time (mins)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'Model': model_name,
            'Language': language,
            'Characters': num_characters,
            'Tokens': num_tokens,
            'Ratio': ratio,
            'Entropy': entropy,
            'Compression Ratio': compression_ratio,
            'Redundancy': redundancy,
            'Time (mins)': time_taken
        })
    logging.info(f'Results saved to {csv_filename}')
    
def AC_compress_file(model, model_name, tokenizer, in_filename, out_filename, max_tokens=None):
    """
    Compress a file using Arithmetic Coding.

    Args:
        model: The language model to use for compression.
        model_name (str): Name of the model.
        tokenizer: The tokenizer associated with the model.
        in_filename (str): Path to the input file.
        out_filename (str): Path to save the compressed file.
        max_tokens (int, optional): Maximum number of tokens to process.
    """
    logging.info(f'Starting compression for file: {in_filename} using model: {model_name}')
    start_time = time.time()

    language = os.path.splitext(os.path.basename(in_filename))[0]
    with open(in_filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    if max_tokens is not None:
        tokens = tokenizer.encode(text)[:max_tokens] 
        logging.info(f'Tokenization truncated to {max_tokens} tokens.')
    else:
        tokens = tokenizer.encode(text)
    
    num_characters = len(tokenizer.batch_decode(tokens, skip_special_tokens=True)) 
    num_tokens = len(tokens)
    ratio = round(num_characters / num_tokens, 2)
    logging.info(f'Number of characters: {num_characters}, Number of tokens: {num_tokens}, Ratio: {ratio}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    probs = []
    encoder = constriction.stream.queue.RangeEncoder()

    input_ids = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64, device=device).unsqueeze(0)
    logging.info(f'Initial input shape: {input_ids.shape}')

    for i in tqdm(range(len(tokens))):
        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.logits[0, -1]
        probabilities = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float64)
        probs.append(probabilities[tokens[i]])

        entropy_model = constriction.stream.model.Categorical(probabilities)
        encoder.encode(tokens[i], entropy_model)

        input_ids = torch.cat((input_ids, torch.tensor([[tokens[i]]], dtype=torch.int64).to(device)), dim=1)
        logging.debug(f'Processed token {i}/{len(tokens)}: {tokens[i]}')

    probs_np = np.array(probs)    
    entropy = (np.sum(-1 * np.log2(probs_np)).item()) / num_characters
    relative_entropy = (len(probs_np) * (-1 * np.log2(1 / len(tokenizer.get_vocab())))) / num_characters
    redundancy = 1 - (entropy / relative_entropy)

    compressed = encoder.get_compressed()
    logging.info(f'Compression completed. Compressed data size: {len(compressed)} bytes.')

    if sys.byteorder != "little":
        compressed.byteswap(inplace=True)

    with open(out_filename, 'wb') as f:
        compressed.tofile(f)
    
    compression_ratio = 8 * os.path.getsize(out_filename) / num_characters
    time_taken = (time.time() - start_time) / 60  # Time in minutes
    
    save_to_csv(model_name, language, num_characters, num_tokens, ratio, entropy, compression_ratio, redundancy, time_taken)

    logging.info(f'Compressed data written to "{out_filename}". Compression ratio: {compression_ratio:.2f}. Time taken: {time_taken:.2f} minutes.')
    
def AC_decompress_file(model, tokenizer, in_filename, out_filename, num_tokens):
    """
    Decompress a file compressed using Arithmetic Coding.

    Args:
        model: The language model to use for decompression.
        tokenizer: The tokenizer associated with the model.
        in_filename (str): Path to the compressed input file.
        out_filename (str): Path to save the decompressed file.
        num_tokens (int): Number of tokens to decompress.
    """
    logging.info(f'Starting decompression for file: {in_filename}')
    start_time = time.time()

    compressed = np.fromfile(in_filename, dtype=np.uint32)
    if sys.byteorder != "little":
        compressed.byteswap(inplace=True)  # Restores native byte order ("endianness")
    decoder = constriction.stream.queue.RangeDecoder(compressed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokens = []

    with open(out_filename, "w", encoding="utf-8") as out_file:
        input_ids = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64, device=device).unsqueeze(0)

        for i in tqdm(range(num_tokens)):
            with torch.no_grad():
                outputs = model(input_ids)

            logits = outputs.logits[0, -1]
            probabilities = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float64)
            entropy_model = constriction.stream.model.Categorical(probabilities)

            next_token_id = decoder.decode(entropy_model)
            tokens.append(next_token_id)
            input_ids = torch.cat((input_ids, torch.tensor([[next_token_id]], dtype=torch.int64).to(device)), dim=1)

            logging.debug(f'Decompressed token {i}/{num_tokens}: {next_token_id}')

        text = tokenizer.batch_decode([tokens], skip_special_tokens=True)
        text = "".join(text)
        out_file.write(text)

    time_taken = (time.time() - start_time) / 60  # Time in minutes
    logging.info(f'Decompression complete. Decompressed data written to "{out_filename}". Time taken: {time_taken:.2f} minutes.')