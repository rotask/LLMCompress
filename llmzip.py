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
        
    def tokenize(self, text, language):
        
        # Log tokenization process
        logging.info(f"Tokenizing text for: {language} using model: {self.model_name} with context size: {Config.CONTEXT_SIZE}")
        
        tokens_path = os.path.join(Config.OUTPUT_DIR, f"Tokens{language}Text{self.model_name}.txt")
        
        if os.path.exists(tokens_path):
            # If tokens file already exists, load tokens from file
            logging.info(f"Loading tokens from file: {tokens_path}")
            with open(tokens_path, 'r', encoding='utf-8') as file:
                token_str = file.read().strip()
                token_ids = list(map(int, token_str.split(',')))
                tokens = torch.tensor(token_ids)
        else:
            # If tokens file does not exist, create new tokens and save to file
            logging.info(f"Creating new tokens and saving to file: {tokens_path}")
            tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze()
            with open(tokens_path, 'w', encoding='utf-8') as file:
                file.write(','.join(map(str, tokens.tolist())))
        
        return tokens
    
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
            padding = torch.tensor([value] * pad_amount)
            tokens = torch.cat((tokens, padding))
        else:
            pad_amount = 0

        return tokens, pad_amount
    
    def forward(self, tokens, index, summary_tokens):
        logging.info(f"Forward method called with tokens shape: {tokens.shape}, index: {index}, summary_tokens shape: {summary_tokens.shape}")
        
        with torch.no_grad():
            batch_size = tokens.shape[0]
            logging.info(f"Batch size: {batch_size}")
            
            # Use summary as side information
            context = torch.cat((summary_tokens.repeat(batch_size, 1), tokens[:, :index]), dim=1)
            logging.info(f"Context shape after concatenation: {context.shape}")
            
            inputs = {'input_ids': context}
            logging.info(f"Model input shape: {inputs['input_ids'].shape}")
            
            try:
                output = self.model(**inputs)
                logits = output.logits[:, -1, :]  # Only take the last token's logits
                logging.info(f"Output logits shape: {logits.shape}")
                return logits
            except RuntimeError as e:
                logging.error(f"RuntimeError in forward method: {str(e)}")
                logging.error(f"Last successful shape - Context: {context.shape}, Inputs: {inputs['input_ids'].shape}")
                raise
    
    def calculate_entropy(self, no_characters, probs):
        entropy = (torch.sum(-1 * torch.log2(probs)).item()) / no_characters
        vocab_size = len(self.tokenizer.get_vocab())
        relative_entropy = (len(probs) * (-1 * np.log2(1 / vocab_size))) / no_characters
        redundancy = 1 - (entropy / relative_entropy)
        return entropy, redundancy

    def zip(self, input_path, output_path, summary, gpu_name):
        start_time = time.time()
        logging.info(f"Starting zipping process for file: {input_path}")
        logging.info(f"Using GPU: {gpu_name}")

        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        num_characters = len(text)
        language = os.path.splitext(os.path.basename(input_path))[0]

        tokens = self.tokenize(text, language)
        summary_tokens = self.tokenizer(summary, return_tensors="pt")["input_ids"].to(self.device)
        summary_length = summary_tokens.shape[1]

        logging.info(f"Tokens shape: {tokens.shape}, Summary tokens shape: {summary_tokens.shape}")

        num_tokens = len(tokens)
        char_per_token = f"{num_characters / num_tokens:.2f}"

        logging.info(f"Tokenization complete. Number of tokens: {num_tokens}, Characters per token: {char_per_token}")

        tokens, pad_amount = self.pad(tokens, self.tokenizer.eos_token_id)
        tokens = tokens.reshape(-1, Config.CONTEXT_SIZE)
        tokens = tokens.to(self.device)

        logging.info(f"Tokens shape after padding and reshaping: {tokens.shape}")

        ranks = torch.zeros((tokens.shape[0], Config.CONTEXT_SIZE), dtype=torch.int32)
        probs = torch.zeros((tokens.shape[0], Config.CONTEXT_SIZE))

        logging.info(f"Ranks shape: {ranks.shape}, Probs shape: {probs.shape}")

        batches = tokens.shape[0] // Config.BATCH_SIZE
        if tokens.shape[0] % Config.BATCH_SIZE != 0:
            batches += 1

        logging.info(f"Total batches: {batches}")

        tqdm_callback = TqdmToLogger(logging.getLogger(), level=logging.INFO)
        for i in tqdm(range(batches), desc="Processing batches", unit="batch", leave=True, file=sys.stdout, dynamic_ncols=True):
            batch = tokens[i*Config.BATCH_SIZE:(i+1)*Config.BATCH_SIZE]
            logging.info(f"Batch {i+1} shape: {batch.shape}")

            curr_ranks = torch.zeros((batch.shape[0], Config.CONTEXT_SIZE), dtype=torch.int32)
            curr_probs = torch.zeros((batch.shape[0], Config.CONTEXT_SIZE))

            for j in tqdm(range(Config.CONTEXT_SIZE), desc="Processing tokens", unit="token", leave=False, file=sys.stdout, dynamic_ncols=True):
                if j % 100 == 0:
                    tqdm_callback(f"Processing token {j+1} of {Config.CONTEXT_SIZE} in batch {i+1}")

                try:
                    logits = self.forward(batch, j, summary_tokens)
                    logging.info(f"Logits shape: {logits.shape}")

                    logits, sorted_tokens = torch.sort(logits, descending=True)
                    probabilities = F.softmax(logits, dim=-1)

                    next_tokens = batch[:, j]
                    next_tokens_expanded = next_tokens.view(-1, 1).expand_as(sorted_tokens)
                    rank_indices = (sorted_tokens == next_tokens_expanded).nonzero(as_tuple=True)[1]
                    curr_ranks[:, j] = rank_indices
                    curr_probs[:, j] = probabilities.gather(1, rank_indices.view(-1, 1)).squeeze()
                except Exception as e:
                    logging.error(f"Error processing token {j} in batch {i}: {str(e)}")
                    logging.error(f"Current shapes - Batch: {batch.shape}, Summary tokens: {summary_tokens.shape}")
                    raise

            ranks[i*Config.BATCH_SIZE:(i+1)*Config.BATCH_SIZE] = curr_ranks
            probs[i*Config.BATCH_SIZE:(i+1)*Config.BATCH_SIZE] = curr_probs

        # Remove padding
        if pad_amount > 0:
            logging.info(f"Removing {pad_amount} padding tokens")
            ranks = ranks[:, :-pad_amount]
            probs = probs[:, :-pad_amount]

        logging.info(f"Final ranks shape: {ranks.shape}, Final probs shape: {probs.shape}")

        ranks = ranks.flatten().int()
        probs = probs.flatten()
        probs = torch.where(probs == 0, probs + 0.001, probs)

        entropy, redundancy = self.calculate_entropy(num_characters, probs)
        logging.info(f"Entropy: {entropy}, Redundancy: {redundancy}")

        zipped_ranks = compress_ranks(ranks)

        with open(output_path, "wb") as file:
            file.write(zipped_ranks)
            logging.info(f"Compression complete! Saved file as {output_path}")

        compression_ratio = os.path.getsize(output_path) * 8 / num_characters
        time_taken = (time.time() - start_time) / 60

        self.save_results(language, num_characters, num_tokens, char_per_token, entropy, compression_ratio, redundancy, time_taken, gpu_name)
        logging.info(f"Zipping process for {input_path} completed successfully in {time_taken:.2f} minutes.")
        
    def unzip(self, input_path, output_path, summary):
        """
        Decompress the data from a zip file and save the decompressed text.

        Args:
            input_path (str): Path to the compressed input file.
            output_path (str): Path to the output decompressed file.
            summary (str): The summary used during compression.
        """
        logging.info(f"Starting Unzipping process for file: {input_path}")
        
        if self.compression_method == "Ranks":
            with open(input_path, "rb") as file:
                zipped_ranks = file.read()
                
            start_time = time.time()
            ranks = decompress_ranks(zipped_ranks)
            
            summary_tokens = self.tokenizer(summary, return_tensors="pt")["input_ids"].to(self.device)
            
            with torch.no_grad():
                # Pad ranks and reshape
                ranks, pad_amount = self.pad(ranks, -999)
                ranks = ranks.reshape(-1, Config.CONTEXT_SIZE)

                # Initialize tokens tensor
                tokens = torch.zeros(ranks.shape, dtype=int).to(self.device)

                # Determine the number of batches
                batches = tokens.shape[0] // Config.BATCH_SIZE
                if tokens.shape[0] % Config.BATCH_SIZE != 0:
                    batches += 1

                logging.info("Getting Tokens...")
                tqdm_callback = TqdmToLogger(logging.getLogger(), level=logging.INFO)
                
                for i in tqdm(range(batches), desc="Processing batches", unit="batch", leave=True, file=sys.stdout, dynamic_ncols=True):
                    curr_ranks = ranks[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE]
                    batch = tokens[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE].to(self.device)
                    tqdm_callback(f"Processing batch {i+1} of {batches}")

                    for j in tqdm(range(Config.CONTEXT_SIZE), desc="Processing tokens", unit="token", leave=False, file=sys.stdout, dynamic_ncols=True):    
                        if j % 100 == 0:
                            tqdm_callback(f"Processing token {j+1} of {Config.CONTEXT_SIZE} in batch {i+1}")
                            
                        logits = self.forward(batch, j, summary_tokens)
                        logits, sorted_tokens = torch.sort(logits, descending=True)
                        indices = curr_ranks[:, j].clone()
                        mask = indices == -999
                        valid_indices = torch.where(mask, torch.tensor(0, device=indices.device), indices)
                        decoded_tokens = sorted_tokens[torch.arange(indices.shape[0]), valid_indices]
                        decoded_tokens[mask] = self.tokenizer.eos_token_id
                        batch[:, j] = decoded_tokens.int()
                    tokens[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE] = batch

                # Flatten tokens and remove padding if necessary
                tokens = tokens.int().flatten()
                if pad_amount != 0:
                    tokens = tokens[:-pad_amount]
                tokens = tokens.reshape((1, -1))

                # Decode tokens to text
                text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                text = "".join(text)

                # Save the decompressed text
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                    
                time_taken = (time.time() - start_time) / 60
                logging.info(f"Decompression Completed in {time_taken:.2f} minutes! Saved as {output_path}")
        else:
            raise ValueError(f"Invalid compression method: {self.compression_method}")
            
    def check(self, original_text, unzipped_text):
        """
        Check if the original text and the unzipped text are the same.

        Args:
            original_text (str): Path to the original text file.
            unzipped_text (str): Path to the unzipped text file.

        Returns:
            bool: True if the texts are the same, False otherwise.
        """
        try:
            with open(original_text, 'r', encoding='utf-8') as original, open(unzipped_text, 'r', encoding='utf-8') as unzipped:
                before = original.read()
                after = unzipped.read()
                
                # Remove all whitespace and convert to lowercase for comparison
                before_cleaned = ''.join(before.split()).lower()
                after_cleaned = ''.join(after.split()).lower()
                
                if before_cleaned == after_cleaned:
                    logging.info("Files MATCH - Zipping and Unzipping was done successfully!")
                    return True
                else:
                    logging.error("The two text files are not the same! There was an error in the process!")
                    logging.error(f"Original text length: {len(before_cleaned)}, Unzipped text length: {len(after_cleaned)}")
                    
                    # Find the first difference
                    for i, (c1, c2) in enumerate(zip(before_cleaned, after_cleaned)):
                        if c1 != c2:
                            logging.error(f"First difference at position {i}: '{c1}' vs '{c2}'")
                            break
                    
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
