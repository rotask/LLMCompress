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

class LLMCompress:
    """
    A class for compressing and decompressing text using language models.
    """

    def __init__(self, model_name, compression_method):
        """
        Initialize the LLMCompress object.

        Args:
            model_name (str): Name of the language model to use.
            compression_method (str): Compression method to use ('Ranks' or 'AC').
        """
        self.model_name = model_name
        self.compression_method = compression_method
        self.model, self.tokenizer = get_model_and_tokenizer(model_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logging.info(f"Initialized LLMCompress with model: {self.model_name} and compression method: {self.compression_method}")
        
    def tokenize(self, text, language):
        """
        Tokenize the input text.

        Args:
            text (str): Input text to tokenize.
            language (str): Language of the input text.

        Returns:
            torch.Tensor: Tokenized text as a tensor of token IDs.
        """
        logging.info(f"Tokenizing text for language: {language} using model: {self.model_name}")
        
        tokens_path = os.path.join(Config.OUTPUT_DIR, f"Tokens{language}Text{self.model_name}.txt")
        
        if os.path.exists(tokens_path):
            logging.info(f"Loading tokens from file: {tokens_path}")
            with open(tokens_path, 'r', encoding='utf-8') as file:
                token_str = file.read().strip()
                token_ids = list(map(int, token_str.split(',')))
                tokens = torch.tensor(token_ids)
        else:
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
            tuple: (padded_tokens, pad_amount)
        """
        remainder = len(tokens) % Config.CONTEXT_SIZE
        
        if remainder > 0:
            pad_amount = Config.CONTEXT_SIZE - remainder
            logging.info(f"Padding amount: {pad_amount}")
            padding = torch.tensor([value] * pad_amount)
            tokens = torch.cat((tokens, padding))
        else:
            pad_amount = 0

        return tokens, pad_amount
    
    def forward(self, tokens, index, past=None):
        """
        Perform a forward pass through the model.

        Args:
            tokens (torch.Tensor): Input tokens.
            index (int): Current index in the sequence.
            past (tuple, optional): Past key/value states for attention.

        Returns:
            tuple: (logits, past_key_values)
        """
        with torch.no_grad():
            inputs = {'input_ids': tokens[:, index].reshape(-1, 1)}
            output = self.model(**inputs, past_key_values=past)
            logits = output.logits
            if len(logits.shape) > 2:
                logits = logits.reshape((logits.shape[0], -1))
            return logits, output.past_key_values
    
    def calculate_entropy(self, no_characters, probs):
        """
        Calculate entropy and redundancy of the compressed data.

        Args:
            no_characters (int): Number of characters in the original text.
            probs (torch.Tensor): Probabilities of each token.

        Returns:
            tuple: (entropy, redundancy)
        """
        entropy = (torch.sum(-1 * torch.log2(probs)).item()) / no_characters
        vocab_size = len(self.tokenizer.get_vocab())
        relative_entropy = (len(probs) * (-1 * np.log2(1 / vocab_size))) / no_characters
        redundancy = 1 - (entropy / relative_entropy)
        return entropy, redundancy
    
    def zip(self, input_path, output_path):
        """
        Compress the text file at the given input path and save the compressed file.

        Args:
            input_path (str): Path to the input text file.
            output_path (str): Path to save the compressed file.
        """
        logging.info(f"Starting compression process for file: {input_path}")
        
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        logging.info(f"Using device: {gpu_name}")
                    
        if self.compression_method == "AC":
            AC_compress_file(self.model, self.model_name, self.tokenizer, input_path, output_path, 500)
        elif self.compression_method == "Ranks":
            # Ranks compression method
            with open(input_path, encoding="utf-8") as f:
                text = f.read()    
                
            start_time = time.time()
            num_characters = len(text)
            language = os.path.splitext(os.path.basename(input_path))[0]
            
            tokens = self.tokenize(text, language)
            num_tokens = len(tokens)
            char_per_token = f"{num_characters / num_tokens:.2f}"
            
            logging.info(f"Tokenization complete. Number of tokens: {num_tokens}, Characters per token: {char_per_token}")
            
            # Padding
            tokens, pad_amount = self.pad(tokens, self.tokenizer.eos_token_id)
            tokens = tokens.reshape(-1, Config.CONTEXT_SIZE)
            
            # Initialize tensors for storing ranks and probabilities
            ranks = torch.zeros(tokens.shape)
            probs = torch.zeros(tokens.shape)
            
            # Add EOS token to the beginning of each sequence
            eos = torch.tensor([self.tokenizer.eos_token_id]*tokens.shape[0]).unsqueeze(1)
            tokens = torch.cat((eos, tokens), 1)
            tokens = tokens.to(self.device)
            
            # Calculate ranks and probabilities
            batches = tokens.shape[0] // Config.BATCH_SIZE
            if tokens.shape[0] % Config.BATCH_SIZE != 0:
                batches += 1
            
            logging.info("Calculating ranks and probabilities...")
            tqdm_callback = TqdmToLogger(logging.getLogger(), level=logging.INFO)
            for i in tqdm(range(batches), desc="Processing batches", unit="batch", leave=True, file=sys.stdout, dynamic_ncols=True):
                batch = tokens[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE]
                curr_ranks = torch.zeros((batch.shape[0], batch.shape[1]-1))
                curr_probs = torch.zeros((batch.shape[0], batch.shape[1]-1))
                past = None
                tqdm_callback(f"Processing batch {i+1} of {batches}")
                
                for j in tqdm(range(batch.shape[1]-1), desc="Processing tokens", unit="token", leave=False, file=sys.stdout, dynamic_ncols=True):
                    if j % 100 == 0:
                        tqdm_callback(f"Processing token {j+1} of {batch.shape[1]-1} in batch {i+1}")
                    
                    logits, past = self.forward(batch, j, past)
                    logits, sorted_tokens = torch.sort(logits, descending=True)
                    probabilities = F.softmax(logits, dim=-1)
                    
                    next_tokens = batch[:, j + 1]
                    next_tokens_expanded = next_tokens.view(-1, 1).expand_as(sorted_tokens)
                    rank_indices = (sorted_tokens == next_tokens_expanded).nonzero(as_tuple=True)
                    
                    rank_indices = rank_indices[1]  # remove index column
                    
                    curr_ranks[:, j] = rank_indices
                    curr_probs[:, j] = probabilities.gather(1, rank_indices.view(-1, 1)).squeeze()
                    
                ranks[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE] = curr_ranks
                probs[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE] = curr_probs
            
            # Flatten and process ranks and probabilities
            ranks = ranks.flatten().int()
            probs = probs.flatten()
            probs = torch.where(probs == 0, probs + 0.001, probs)
            logging.info(f"Ranks and probabilities calculation complete. Total ranks: {len(ranks)}, Total probabilities: {len(probs)}")
            
            # Remove padding if necessary
            if pad_amount > 0:
                ranks = ranks[:-pad_amount]
                probs = probs[:-pad_amount]
            
            # Calculate entropy and redundancy
            entropy, redundancy = self.calculate_entropy(num_characters, probs)
            logging.info(f"Entropy: {entropy}, Redundancy: {redundancy}")

            # Save probabilities
            probs_filename = f"Probabilities{self.model_name}{language}.txt"
            probs_path = os.path.join("Output_Files", probs_filename)
            with open(probs_path, 'w', encoding='utf-8') as file:
                file.write(','.join(map(str, probs.tolist())))
            
            # Compress ranks
            zipped_ranks = compress_ranks(ranks)
            
            # Save compressed data
            with open(output_path, "wb") as file:
                file.write(zipped_ranks)
                logging.info(f"Compression complete! Saved file as {output_path}")
            
            compression_ratio = os.path.getsize(output_path) * 8 / num_characters
            time_taken = (time.time() - start_time) / 60  # Calculate time in minutes
            
            self.save_results(language, num_characters, num_tokens, char_per_token, entropy, compression_ratio, redundancy, time_taken, gpu_name)
            logging.info(f"Compression process for {input_path} completed successfully in {time_taken:.2f} minutes.")
            
        else:
            raise ValueError(f"Invalid compression method: {self.compression_method}")
        
    def unzip(self, input_path, output_path):
        """
        Decompress the data from a compressed file and save the decompressed text.

        Args:
            input_path (str): Path to the compressed input file.
            output_path (str): Path to save the decompressed output file.
        """
        logging.info(f"Starting decompression process for file: {input_path}")
        
        if self.compression_method == "AC":
            AC_decompress_file(self.model, self.tokenizer, input_path, output_path, 500)
        elif self.compression_method == "Ranks":
            # Ranks decompression method
            with open(input_path, "rb") as file:
                zipped_ranks = file.read()
                
            start_time = time.time()
            ranks = decompress_ranks(zipped_ranks)
            
            with torch.no_grad():
                # Pad ranks and reshape
                ranks, pad_amount = self.pad(ranks, -999)
                ranks = ranks.reshape(-1, Config.CONTEXT_SIZE)

                # Initialize tokens tensor
                tokens = torch.zeros(ranks.shape, dtype=int).to(self.device)
                eos = torch.full((tokens.shape[0], 1), self.tokenizer.eos_token_id, dtype=tokens.dtype, device=self.device)
                tokens = torch.cat((eos, tokens), 1)

                # Determine the number of batches
                batches = tokens.shape[0] // Config.BATCH_SIZE
                if tokens.shape[0] % Config.BATCH_SIZE != 0:
                    batches += 1

                logging.info("Reconstructing tokens...")
                tqdm_callback = TqdmToLogger(logging.getLogger(), level=logging.INFO)
                
                for i in tqdm(range(batches), desc="Processing batches", unit="batch", leave=True, file=sys.stdout, dynamic_ncols=True):
                    curr_ranks = ranks[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE]
                    batch = tokens[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE].to(self.device)
                    past = None
                    tqdm_callback(f"Processing batch {i+1} of {batches}")

                    for j in tqdm(range(Config.CONTEXT_SIZE), desc="Processing tokens", unit="token", leave=False, file=sys.stdout, dynamic_ncols=True):    
                        if j % 100 == 0:
                            tqdm_callback(f"Processing token {j+1} of {Config.CONTEXT_SIZE - 1} in batch {i+1}")
                            
                        logits, past = self.forward(batch, j, past)
                        logits, sorted_tokens = torch.sort(logits, descending=True)
                        indices = curr_ranks[:, j].clone()
                        mask = indices == -999
                        valid_indices = torch.where(mask, torch.tensor(0, device=indices.device), indices)
                        decoded_tokens = sorted_tokens[torch.arange(indices.shape[0]), valid_indices]
                        decoded_tokens[mask] = self.tokenizer.eos_token_id
                        batch[:, j + 1] = decoded_tokens.int()
                    tokens[i*Config.BATCH_SIZE:(i + 1)*Config.BATCH_SIZE] = batch

                # Flatten tokens and remove padding if necessary
                tokens = tokens[:, 1:].int().flatten()
                if pad_amount != 0:
                    tokens = tokens[:-pad_amount]
                tokens = tokens.reshape((1, -1))

                # Decode tokens to text
                text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                text = "".join(text)

                # Save the decompressed text
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                    
                time_taken = (time.time() - start_time) / 60  # Calculate time in minutes
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
                if before == after:
                    logging.info("Files MATCH - Compression and Decompression was done successfully!")
                    return True
                else:
                    logging.error("The two text files are not the same! There was an error in the process!")
                    return False
        except FileNotFoundError:
            logging.error("One of the files was not found.")
            return False
        except IOError as e:
            logging.error(f"An error occurred while reading the files: {e}")
            return False
        
    def save_results(self, text_file, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy, time_taken, gpu_name):
        """
        Save the compression results to a CSV file.

        Args:
            text_file (str): Text file compressed.
            total_chars (int): Total number of characters in the original text.
            token_length (int): Number of tokens in the tokenized text.
            char_token_ratio (float): Ratio of characters to tokens.
            entropy (float): Calculated entropy of the compressed data.
            compression_ratio (float): Achieved compression ratio.
            redundancy (float): Calculated redundancy.
            time_taken (float): Time taken for compression in minutes.
            gpu_name (str): Name of the GPU used for compression.
        """
        csv_path = os.path.join(Config.RESULTS_DIR, f'{self.model_name}_data_{self.compression_method}.csv')
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['GPU Used','Text File', 'Total Characters', 'Total Tokens', 'Characters/Tokens', 'Entropy', 'Compression Ratio', 'Redundancy', 'Time(mins)'])
            writer.writerow([gpu_name, text_file, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy, time_taken])