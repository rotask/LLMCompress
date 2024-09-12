import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import csv
import sys
from config import Config
from logger import setup_logging, TqdmToLogger
from models import get_model_and_tokenizer
from compression import compress_ranks, decompress_ranks
from Arithmetic_Coder import AC_compress_file, AC_decompress_file
import logging
import time


class LLMZip:
    def __init__(self, model_name, compression_method):
        self.model_name = model_name
        self.compression_method = compression_method
        self.model, self.tokenizer = get_model_and_tokenizer(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Log initialization
        logging.info(f"Initialized LLMZip with model: {self.model_name} and compression method: {self.compression_method}")
        
    def tokenize(self, text, prefix):
        tokens_path = os.path.join(Config.OUTPUT_DIR, f"Tokens_{prefix}_{self.model_name}.txt")
        
        if os.path.exists(tokens_path):
            logging.info(f"Loading tokens from file: {tokens_path}")
            with open(tokens_path, 'r', encoding='utf-8') as file:
                token_str = file.read().strip()
                token_ids = list(map(int, token_str.split(',')))
                tokens = torch.tensor(token_ids)
        else:
            logging.info(f"Creating new tokens and saving to file: {tokens_path}")
            tokens = self.tokenizer.encode(text, return_tensors="pt").squeeze()
            with open(tokens_path, 'w', encoding='utf-8') as file:
                file.write(','.join(map(str, tokens.tolist())))
        
        logging.info(f"Number of tokens for {prefix}: {len(tokens)}")
        return tokens.to(self.device)

    
    def pad(self, tokens, value):
        """
        Pad the tokens to make their length a multiple of the context size.

        Args:
            tokens (torch.Tensor): The tensor of token IDs.
            value (int): The padding value.

        Returns:
            torch.Tensor: The padded tokens.
            int: The amount of padding added.
        """
        # Calculate the remainder to determine how much padding is needed
        remainder = len(tokens) % Config.CONTEXT_SIZE
        
        if remainder > 0:
            pad_amount = Config.CONTEXT_SIZE - remainder
            logging.info(f"Padding amount: {pad_amount}")
            # Create padding tensor on the same device as tokens
            padding = torch.full((pad_amount,), value, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat((tokens, padding))
        else:
            pad_amount = 0

        return tokens, pad_amount
    
    def forward(self, tokens, index, summary_tokens):
        with torch.no_grad():
            batch_size = tokens.shape[0]
            
            # Ensure all tensors are on the same device
            device = tokens.device
            summary_tokens = summary_tokens.to(device)
            
            # Use summary as side information and include predicted tokens up to index
            context = torch.cat((summary_tokens.unsqueeze(0).repeat(batch_size, 1), tokens[:, :index]), dim=1)
            
            # Check if context exceeds model's maximum input length
            max_length = self.model.config.max_position_embeddings
            if context.shape[1] > max_length:
                context = context[:, -max_length:]  # Take the last max_length tokens
            
            inputs = {'input_ids': context}
            
            output = self.model(**inputs)
            logits = output.logits[:, -1, :]  # Only take the last token's logits
            return logits
    
    def calculate_entropy(self, no_characters, probs):
        entropy = (torch.sum(-1 * torch.log2(probs)).item()) / no_characters
        vocab_size = len(self.tokenizer.get_vocab())
        relative_entropy = (len(probs) * (-1 * np.log2(1 / vocab_size))) / no_characters
        redundancy = 1 - (entropy / relative_entropy)
        return entropy, redundancy

    def zip(self, input_path, summary, gpu_name):
        start_time = time.time()
        logging.info(f"Starting zipping process for file: {input_path}")
        logging.info(f"Using GPU: {gpu_name}")

        output_path = Config.get_output_file_path(input_path, self.model_name, self.compression_method)
        logging.info(f"Output will be saved to: {output_path}")

        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        num_characters = len(text)
        logging.info(f"Original text length: {num_characters} characters")

        tokens = self.tokenize(text, "InputText")
        summary_tokens = self.tokenizer(summary, return_tensors="pt")["input_ids"].squeeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = tokens.to(device)
        summary_tokens = summary_tokens.to(device)

        logging.info(f"Tokens shape: {tokens.shape}, Summary tokens shape: {summary_tokens.shape}")

        max_length = self.model.config.max_position_embeddings
        effective_context_size = min(Config.CONTEXT_SIZE, max_length - len(summary_tokens))
        
        logging.info(f"Summary length: {len(summary_tokens)}, Effective context size: {effective_context_size}")

        ranks = []
        context = summary_tokens.clone()

        for i in tqdm(range(len(tokens)), desc="Processing tokens"):
            if len(context) >= max_length:
                context = torch.cat([summary_tokens, context[-(max_length-len(summary_tokens)):]])

            logits = self.model(input_ids=context.unsqueeze(0)).logits[0, -1]
            probabilities = F.softmax(logits, dim=-1)
            sorted_indices = torch.argsort(logits, descending=True)
            rank = (sorted_indices == tokens[i]).nonzero().item()
            
            ranks.append(rank)
            context = torch.cat([context, tokens[i].unsqueeze(0)])
            
            logging.debug(f"Token: {self.tokenizer.decode([tokens[i].item()])}, Rank: {rank}")

        ranks = torch.tensor(ranks, dtype=torch.int32)

        zipped_ranks = compress_ranks(ranks.cpu())

        with open(output_path, "wb") as file:
            file.write(zipped_ranks)
            logging.info(f"Compression complete! Saved file as {output_path}")

        compression_ratio = os.path.getsize(output_path) * 8 / num_characters
        time_taken = (time.time() - start_time) / 60

        logging.info(f"Zipping process for {input_path} completed successfully in {time_taken:.2f} minutes.")
        return compression_ratio

    def unzip(self, input_path, output_path, summary):
        logging.info(f"Starting Unzipping process for file: {input_path}")

        with open(input_path, "rb") as file:
            zipped_ranks = file.read()
        logging.info(f"Read {len(zipped_ranks)} bytes from compressed file")

        start_time = time.time()
        ranks = decompress_ranks(zipped_ranks)
        logging.info(f"Decompressed ranks shape: {ranks.shape}")

        summary_tokens = self.tokenizer(summary, return_tensors="pt")["input_ids"].squeeze(0)
        logging.info(f"Summary tokens shape: {summary_tokens.shape}")

        device = next(self.model.parameters()).device
        summary_tokens = summary_tokens.to(device)
        ranks = ranks.to(device)

        max_length = self.model.config.max_position_embeddings
        effective_context_size = min(Config.CONTEXT_SIZE, max_length - len(summary_tokens))

        decoded_tokens = []
        context = summary_tokens.clone()

        with torch.no_grad():
            for i in tqdm(range(len(ranks)), desc="Processing ranks"):
                if len(context) >= max_length:
                    context = torch.cat([summary_tokens, context[-(max_length-len(summary_tokens)):]])

                logits = self.model(input_ids=context.unsqueeze(0)).logits[0, -1]
                sorted_indices = torch.argsort(logits, descending=True)
                decoded_token = sorted_indices[ranks[i]].item()
                
                decoded_tokens.append(decoded_token)
                context = torch.cat([context, torch.tensor([decoded_token], device=device)])
                
                logging.debug(f"Rank: {ranks[i].item()}, Decoded token: {self.tokenizer.decode([decoded_token])}")

        logging.info("Token reconstruction completed. Decoding text...")
        text = self.tokenizer.decode(decoded_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logging.info(f"Decoded text length: {len(text)}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        time_taken = (time.time() - start_time) / 60
        logging.info(f"Decompression Completed in {time_taken:.2f} minutes! Saved as {output_path}")

        # Log a sample of the decompressed text
        sample_length = min(1000, len(text))
        logging.info(f"Sample of decompressed text (first {sample_length} characters):")
        logging.info(text[:sample_length])
                
    def check(self, original_text, unzipped_text):
        try:
            with open(original_text, 'r', encoding='utf-8') as original, open(unzipped_text, 'r', encoding='utf-8') as unzipped:
                before = original.read()
                after = unzipped.read()
                
                if before == after:
                    logging.info("Files MATCH - Zipping and Unzipping was done successfully!")
                    return True
                else:
                    logging.error("The two text files are not the same! There was an error in the process!")
                    
                    # Detailed comparison
                    min_len = min(len(before), len(after))
                    for i in range(min_len):
                        if before[i] != after[i]:
                            context = 10  # Show 10 characters before and after the mismatch
                            start = max(0, i - context)
                            end = min(min_len, i + context + 1)
                            logging.error(f"Mismatch at position {i}:")
                            logging.error(f"Original: ...{before[start:end]}...")
                            logging.error(f"Unzipped: ...{after[start:end]}...")
                            break

                    if len(before) != len(after):
                        logging.error(f"Length mismatch: Original {len(before)}, Unzipped {len(after)}")

                    return False
        except FileNotFoundError:
            logging.error("One of the files was not found.")
            return False
        except IOError as e:
            logging.error(f"An error occurred while reading the files: {e}")
            return False
                    
    def save_results(self, language, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy, time_taken, gpu_name):
            csv_path = os.path.join(Config.RESULTS_DIR, f'{self.model_name}_data_{self.compression_method}.csv')
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['GPU Used', 'Text File', 'Context Size', 'Batch Size', 'Total Characters', 'Total Tokens', 'Characters/Tokens', 'Entropy', 'Compression Ratio', 'Redundancy', 'Time(mins)'])
                writer.writerow([gpu_name, language, Config.CONTEXT_SIZE, Config.BATCH_SIZE, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy, time_taken])
