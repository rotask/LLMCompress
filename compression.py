import torch
import numpy as np
import zlib
import logging

def compress_ranks(ranks):
    """
    Compresses the given ranks using zlib compression and handles overflow values.

    Args:
        ranks (torch.Tensor): The ranks to be compressed.

    Returns:
        bytes: The zlib-compressed ranks.

    Note:
        This function uses int32 to handle larger values and logs a warning
        if any rank exceeds the maximum value of uint16 (65535).
    """
    # Convert ranks to numpy array with int32 dtype
    ranks_numpy = ranks.numpy().astype(np.int32)

    # Check for overflow (values larger than uint16 max)
    max_uint16 = 65535
    overflow_mask = ranks_numpy > max_uint16
    overflow_values = ranks_numpy[overflow_mask]

    if len(overflow_values) > 0:
        logging.warning(f"Warning: {len(overflow_values)} ranks exceed the maximum limit of {max_uint16}")
        logging.warning(f"Overflow values: {overflow_values}")

    # Compress using zlib with maximum compression level
    zipped_ranks = zlib.compress(ranks_numpy.tobytes(), level=9)

    return zipped_ranks

def decompress_ranks(zipped_ranks):
    """
    Decompresses the zlib compressed data and converts it into a PyTorch tensor.

    Args:
        zipped_ranks (bytes): The zlib compressed data.

    Returns:
        torch.Tensor: A PyTorch tensor containing the decompressed and processed data.
    """
    # Decompress zlib data
    decompressed_data = zlib.decompress(zipped_ranks)
    
    # Convert back to numpy array with int32 type
    ranks_array = np.frombuffer(decompressed_data, dtype=np.int32)
    
    # Make the array writable by creating a copy
    ranks_array = ranks_array.copy()
    
    # Convert to a PyTorch tensor
    ranks_tensor = torch.from_numpy(ranks_array)
    
    return ranks_tensor