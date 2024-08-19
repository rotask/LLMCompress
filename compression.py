import torch
import numpy as np
import zlib

def compress_ranks(ranks):
    """
    Compresses the given ranks using zlib compression and handles overflow values.

    Parameters:
    ranks (torch.Tensor): The ranks to be compressed.

    Returns:
    tuple: A tuple containing the compressed ranks and the overflow map list.
    """

    max_uint16 = 65535
    ranks_numpy = ranks.numpy()

    # Find values that exceed max_uint16
    overflow_mask = ranks_numpy > max_uint16

    # Create a mapping for overflow values
    overflow_values = ranks_numpy[overflow_mask]
    unique_overflow = np.unique(overflow_values)
    overflow_map = {val: i + max_uint16 for i, val in enumerate(unique_overflow)}

    # Apply the mapping
    compressed_ranks = ranks_numpy.copy()
    for original, mapped in overflow_map.items():
        compressed_ranks[ranks_numpy == original] = mapped

    # Convert to uint16
    compressed_ranks = compressed_ranks.astype(np.uint16)

    # Compress using zlib
    zipped_ranks = zlib.compress(compressed_ranks.tobytes(), level=9)

    # Prepare overflow map for storage
    overflow_map_list = [(int(k), int(v)) for k, v in overflow_map.items()]

    return zipped_ranks, overflow_map_list

def decompress_ranks(zipped_ranks, overflow_map_list):
    """
    Decompresses the zlib compressed data and performs additional operations to convert it into a PyTorch tensor.
    Args:
        zipped_ranks (bytes): The zlib compressed data.
        overflow_map_list (list): A list of tuples representing the overflow map.
    Returns:
        torch.Tensor: A PyTorch tensor containing the decompressed and processed data.
    """
    # Decompress zlib data
    decompressed_data = zlib.decompress(zipped_ranks)
    
    # Convert back to numpy array
    ranks_array = np.frombuffer(decompressed_data, dtype=np.uint16)
    
    # Make the array writable by creating a copy
    ranks_array = ranks_array.copy()
    
    # Recreate overflow map
    overflow_map = {v: k for k, v in overflow_map_list}
    
    # Apply reverse mapping
    for mapped, original in overflow_map.items():
        ranks_array[ranks_array == mapped] = original
    
    # Convert to a compatible PyTorch tensor type (e.g., int32)
    ranks_tensor = torch.from_numpy(ranks_array.astype(np.int32))
    
    return ranks_tensor