import sys
import os
import csv
import torch
import constriction
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging


def save_to_csv(model, language, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy):
        csv_path = f'{model}_data_AC.csv'
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(['Language', 'Total Characters', 'Total Tokens', 'Characters/Tokens', 'Entropy', 'Compression Ratio', 'Redundancy'])

            writer.writerow([language, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy])

def AC_compress_file(model, model_name, tokenizer, in_filename, out_filename, max_tokens=None):
    language = os.path.splitext(os.path.basename(in_filename))[0]
    with open(in_filename, 'r', encoding='utf-8') as file:
        text = file.read()
    
    if max_tokens is not None:
        tokens = tokenizer.encode(text)[:max_tokens] 
    else:
        tokens = tokenizer.encode(text)
        
    num_characters = len(tokenizer.batch_decode(tokens, skip_special_tokens = True)) 
    num_tokens = len(tokens)
    ratio =  num_characters/num_tokens  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    probs = []
    encoder = constriction.stream.queue.RangeEncoder()
    #print(tokens)
    # print(len(tokens))
    input_ids = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64, device = "cuda:0").unsqueeze(0)
    print("Shape is: " , input_ids.shape)
    for i in tqdm(range(len(tokens))):
            
        # print("Current Token: " , tokens[i])
        # print("Inputs:" , input_ids)

        with torch.no_grad():
            outputs = model(input_ids)


        logits = outputs.logits[0, -1]

        probabilities = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float64)
        probs.append(probabilities[tokens[i]])
        #print("Probabilities: " , probabilities)
        entropy_model = constriction.stream.model.Categorical(probabilities)

        encoder.encode(tokens[i], entropy_model)
        input_ids = torch.cat((input_ids, torch.tensor([[tokens[i]]], dtype=torch.int64).to(device)), dim=1)
        
    probs_np = np.array(probs)    
    entropy = (np.sum(-1*np.log2(probs_np)).item())/num_characters
    relative_entropy = (len(probs_np)*(-1*np.log2(1 / len(tokenizer.get_vocab()))))/num_characters
    redundacy = 1 - (entropy/relative_entropy) 
    
       
    compressed = encoder.get_compressed()
    print(compressed)

    if sys.byteorder != "little":
        compressed.byteswap(inplace=True)

    with open(out_filename, 'wb') as f:
        compressed.tofile(f)
        
    compression_ratio = 8*os.path.getsize(out_filename)/num_characters
    
    save_to_csv(model_name, language,num_characters,num_tokens,ratio,entropy,compression_ratio, redundacy)
    
    logging.info(f'Compressed data written to "{out_filename}".')

def AC_decompress_file(model, tokenizer, in_filename, out_filename, num_tokens):
    compressed = np.fromfile(in_filename, dtype=np.uint32)
    if sys.byteorder != "little":
        compressed.byteswap(inplace=True)  # restores native byte order ("endianness").
    decoder = constriction.stream.queue.RangeDecoder(compressed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = []

    with open(out_filename, "w", encoding="utf-8") as out_file:
        input_ids = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64, device = "cuda:0").unsqueeze(0)
        
        for _ in tqdm(range(num_tokens)):
            with torch.no_grad():
                outputs = model(input_ids)

            logits = outputs.logits[0, -1]
            print(input_ids)
            probabilities = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float64)
            print(probabilities)
            entropy_model = constriction.stream.model.Categorical(probabilities)
            
            next_token_id = decoder.decode(entropy_model)
            print(next_token_id)
            tokens.append(next_token_id)
            

            input_ids = torch.cat((input_ids, torch.tensor([[next_token_id]], dtype=torch.int64).to(device)), dim=1)
        text = (tokenizer.batch_decode([tokens], skip_special_tokens = True))
        text = "".join(text)
        out_file.write(text)
    logging.info(f'Wrote decompressed data to file "{out_filename}".')
    
# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "01-ai/Yi-34B"


# tokenizer = AutoTokenizer.from_pretrained(model_name)

# print("Vocab size is: " , tokenizer.vocab_size)
