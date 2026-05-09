import torch
import numpy as np
import os
import csv
import struct
import sys
import logging
from tqdm import tqdm
import constriction
import time
from config import Config

# Compressed AC files start with a 4-byte little-endian uint32 header carrying
# the original token count. This makes the file self-contained: decompression
# no longer requires the token count to be passed out-of-band.
_HEADER_FORMAT = "<I"
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)


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

    Uses past_key_values for O(N) compute (was O(N^2) before fix). The output
    file begins with a 4-byte little-endian uint32 header carrying the token
    count; the AC payload follows.

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

    # Character count of the actual decoded sequence. Using batch_decode on a
    # flat token list returns a list of N strings (one per token); len() of
    # that gives the token count, not the character count. Use decode() and
    # measure the length of the resulting string instead.
    num_characters = len(tokenizer.decode(tokens, skip_special_tokens=True))
    num_tokens = len(tokens)
    ratio = round(num_characters / num_tokens, 2)
    logging.info(f'Number of characters: {num_characters}, Number of tokens: {num_tokens}, Ratio: {ratio}')

    device = next(model.parameters()).device

    probs = []
    encoder = constriction.stream.queue.RangeEncoder()

    # Feed only the next token at each step; reuse past_key_values for the
    # rest. This keeps total compute O(N) rather than O(N^2).
    input_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.int64, device=device)
    past_key_values = None
    logging.info(f'Initial input shape: {input_id.shape}')

    for i in tqdm(range(len(tokens))):
        with torch.no_grad():
            outputs = model(input_id, past_key_values=past_key_values, use_cache=True)

        logits = outputs.logits[0, -1]
        past_key_values = outputs.past_key_values
        probabilities = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float64)
        probs.append(probabilities[tokens[i]])

        # perfect=True matches constriction <= 0.3 behaviour and ensures that
        # symmetric encoder/decoder default behaviour does not depend on the
        # constriction version; suppresses the deprecation warning.
        entropy_model = constriction.stream.model.Categorical(probabilities, perfect=True)
        encoder.encode(tokens[i], entropy_model)

        # Next iteration: feed only the just-encoded token; KV cache covers the rest.
        input_id = torch.tensor([[tokens[i]]], dtype=torch.int64, device=device)
        logging.debug(f'Processed token {i}/{len(tokens)}: {tokens[i]}')

    probs_np = np.array(probs)
    entropy = (np.sum(-1 * np.log2(probs_np)).item()) / num_characters
    relative_entropy = (len(probs_np) * (-1 * np.log2(1 / len(tokenizer.get_vocab())))) / num_characters
    redundancy = 1 - (entropy / relative_entropy)

    compressed = encoder.get_compressed()
    logging.info(f'Compression completed. Compressed array length: {len(compressed)}.')

    if sys.byteorder != "little":
        compressed.byteswap(inplace=True)

    # Write the length header followed by the AC payload so the file is
    # self-contained for decompression.
    payload_bytes = compressed.tobytes()
    header = struct.pack(_HEADER_FORMAT, num_tokens)
    with open(out_filename, 'wb') as f:
        f.write(header)
        f.write(payload_bytes)

    compression_ratio = 8 * os.path.getsize(out_filename) / num_characters
    time_taken = (time.time() - start_time) / 60  # Time in minutes

    save_to_csv(model_name, language, num_characters, num_tokens, ratio, entropy, compression_ratio, redundancy, time_taken)

    logging.info(f'Compressed data written to "{out_filename}". Compression ratio: {compression_ratio:.2f}. Time taken: {time_taken:.2f} minutes.')


def AC_decompress_file(model, tokenizer, in_filename, out_filename, num_tokens=None):
    """
    Decompress a file compressed using Arithmetic Coding.

    The token count is read from the 4-byte LE header at the start of the
    file. The `num_tokens` parameter is retained for backward compatibility:
    if explicitly provided it overrides the header value.

    Args:
        model: The language model to use for decompression.
        tokenizer: The tokenizer associated with the model.
        in_filename (str): Path to the compressed input file.
        out_filename (str): Path to save the decompressed file.
        num_tokens (int, optional): Override token count. Defaults to None
            (use the header value).
    """
    logging.info(f'Starting decompression for file: {in_filename}')
    start_time = time.time()

    with open(in_filename, 'rb') as f:
        header = f.read(_HEADER_SIZE)
        if len(header) < _HEADER_SIZE:
            raise ValueError(f"Compressed file too short to contain header: {in_filename}")
        header_num_tokens = struct.unpack(_HEADER_FORMAT, header)[0]
        payload_bytes = f.read()

    if num_tokens is None:
        num_tokens = header_num_tokens
    elif num_tokens != header_num_tokens:
        logging.warning(
            f'num_tokens override ({num_tokens}) differs from file header '
            f'({header_num_tokens}). Using override value.'
        )

    # Convert the AC payload (bytes) back to the uint32 array constriction expects.
    compressed = np.frombuffer(payload_bytes, dtype=np.uint32).copy()
    if sys.byteorder != "little":
        compressed.byteswap(inplace=True)
    decoder = constriction.stream.queue.RangeDecoder(compressed)

    device = next(model.parameters()).device

    tokens = []
    input_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.int64, device=device)
    past_key_values = None

    with open(out_filename, "w", encoding="utf-8") as out_file:
        for i in tqdm(range(num_tokens)):
            with torch.no_grad():
                outputs = model(input_id, past_key_values=past_key_values, use_cache=True)

            logits = outputs.logits[0, -1]
            past_key_values = outputs.past_key_values
            probabilities = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float64)
            entropy_model = constriction.stream.model.Categorical(probabilities, perfect=True)

            next_token_id = decoder.decode(entropy_model)
            tokens.append(next_token_id)
            input_id = torch.tensor([[next_token_id]], dtype=torch.int64, device=device)

            logging.debug(f'Decompressed token {i}/{num_tokens}: {next_token_id}')

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        out_file.write(text)

    time_taken = (time.time() - start_time) / 60  # Time in minutes
    logging.info(f'Decompression complete. Decompressed data written to "{out_filename}". Time taken: {time_taken:.2f} minutes.')
