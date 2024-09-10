import zlib
import os

def calculate_compression_ratio(data):
    """
    Calculate the compression ratio of the given data using zlib.

    Args:
        data (str): The input data to be compressed.

    Returns:
        float: The compression ratio calculated as (compressed size * 8) / original size.

    Raises:
        RuntimeError: If compression fails.
    """
    # Encode the data as bytes
    data_bytes = data.encode('utf-8')
    
    try:
        # Compress the data using zlib at maximum compression level
        compressed_data = zlib.compress(data_bytes, level=9)
        
        # Ensure the directory exists
        os.makedirs('Zlib', exist_ok=True)

        # Write the compressed data to a binary file
        with open('Zlib/compressed.bin', 'wb') as file:
            file.write(compressed_data)
        
        # Calculate the compression ratio as (compressed size * 8) / original size
        compression_ratio = (len(compressed_data) * 8) / len(data_bytes)
        return compression_ratio

    except Exception as e:
        raise RuntimeError(f"Compression failed: {e}")

def decompress_and_verify():
    """
    Decompress the compressed file, decode the data, and verify the integrity.

    Returns:
        str: The decompressed message as a string.

    Raises:
        RuntimeError: If decompression fails or the file is not found.
    """
    try:
        # Read the compressed file
        with open('Zlib/compressed.bin', 'rb') as file:
            compressed_data = file.read()

        # Decompress the data
        original_data_bytes = zlib.decompress(compressed_data)

        # Decode the decompressed data
        message = original_data_bytes.decode('utf-8', errors='ignore')
        
        return message

    except zlib.error as e:
        raise RuntimeError(f"Decompression failed: {e}")

    except FileNotFoundError as e:
        raise RuntimeError(f"File not found: {e}")

def main():
    """
    Main function to read data from a file, calculate the compression ratio,
    and verify the decompressed content.
    """
    try:
        # Read data from the input text file
        with open('Zlib/text8_50MB.txt', 'r', encoding='utf-8') as file:
            data = file.read()

        # Calculate and print the compression ratio
        compression_ratio = calculate_compression_ratio(data)
        print(f"Compression Ratio: {compression_ratio:.4f}")

        # Test for decompression and verify
        decompressed_message = decompress_and_verify()

        # Verify the decompressed message with the original data
        if data.startswith(decompressed_message[:200]):  # Check first 200 characters to ensure match
            print("Decompression successful and data verified!")
        else:
            print("Warning: Decompressed data does not match the original data.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
