import os
import re
from datasets import load_dataset

# Load the BookCorpus dataset
dataset = load_dataset("bookcorpus", split="train")

# Define the output file path
output_file_path = "Data/bookcorpus_1MB.txt"

# Number of characters to extract (1MB)
num_chars_to_extract = 1048576
extracted_text = ""

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

# Iterate over the dataset and extract text until the required number of characters is reached
for sample in dataset:
    extracted_text += sample["text"]
    if len(extracted_text) >= num_chars_to_extract:
        extracted_text = extracted_text[:num_chars_to_extract]  # Ensure it's exactly 1MB
        break

# Format the extracted text
formatted_text = format_text(extracted_text)

# Open the output file and write the formatted text to it
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(formatted_text)

print(f"The first {num_chars_to_extract} characters have been extracted, formatted, and saved to {output_file_path}.")

# Check the size of the output file
file_size = os.path.getsize(output_file_path)
print(f"The size of {output_file_path} is {file_size} bytes.")

# Verify if the file size is approximately 1MB
if file_size == num_chars_to_extract:
    print("The file size is correct and matches the expected 1MB.")
else:
    print("The file size does not match the expected 1MB. Please check the file and the extraction process.")
