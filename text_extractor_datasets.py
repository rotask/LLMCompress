import os
import re
from datasets import load_dataset

# Load the BookCorpus dataset
dataset = load_dataset("bookcorpus", split="train")

# Define the output file path
output_file_path = "Data/bookcorpus_1MB.txt"

# Target file size in bytes (1MB)
target_file_size = 1048576

# Overestimate the number of characters to extract (e.g., 10% more)
overestimate_factor = 1.1
num_chars_to_extract = int(target_file_size * overestimate_factor)

def format_text(text):
    # Replace common contractions with their correct form
    text = re.sub(r" n't", "n't", text)
    text = re.sub(r" 's", "'s", text)
    text = re.sub(r" 'd", "'d", text)
    text = re.sub(r" 're", "'re", text)
    text = re.sub(r" 've", "'ve", text)
    text = re.sub(r" 'll", "'ll", text)
    
    # Fix spaces around punctuation
    text = re.sub(r" ,", ",", text)  # Fix spaces before commas
    text = re.sub(r" \.", ".", text)  # Fix spaces before periods
    text = re.sub(r" \?", "?", text)  # Fix spaces before question marks
    text = re.sub(r" \!", "!", text)  # Fix spaces before exclamation marks
    text = re.sub(r"\s+([.,!?;'])", r"\1", text)  # Remove spaces before common punctuation marks
    text = re.sub(r'``\s*', '"', text)  # Replace double backticks with quotes
    text = re.sub(r"\s*''", '"', text)  # Replace trailing backticks with quotes
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space

    # Capitalize the first letter of each sentence
    sentences = re.split(r'(\.|\?|\!)(\s+)', text)
    sentences = [s.capitalize() if not s.endswith(('.', '!', '?')) else s for s in sentences]
    text = ''.join(sentences)
    
    # Ensure the first letter of the text is capitalized
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    
    formatted_text = text
    return formatted_text

# Function to trim text to meet the exact file size
def trim_text_to_size(text, target_size, output_path):
    formatted_text = format_text(text)

    # Write the formatted text to the file
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(formatted_text)

    current_size = os.path.getsize(output_path)

    # If the file size exceeds the target, trim the text
    if current_size > target_size:
        # Calculate how much to trim
        excess_size = current_size - target_size
        # Estimate the number of characters to remove
        trim_length = int((excess_size / current_size) * len(formatted_text))

        # Trim the text
        formatted_text = formatted_text[:-trim_length]

        # Write the trimmed text back to the file
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write(formatted_text)
        
        current_size = os.path.getsize(output_file_path)

    return current_size

# Extract more text initially
extracted_text = ""
for sample in dataset:
    extracted_text += sample["text"]
    if len(extracted_text) >= num_chars_to_extract:
        break

# Trim the text to meet the exact 1MB file size
final_size = trim_text_to_size(extracted_text, target_file_size, output_file_path)

print(f"The text has been extracted, formatted, and adjusted. The final size of {output_file_path} is {final_size} bytes.")

# Verify if the file size is approximately 1MB
if final_size == target_file_size:
    print("The file size is correct and matches the expected 1MB.")
else:
    print("The file size does not match the expected 1MB. Please check the file and the extraction process.")
