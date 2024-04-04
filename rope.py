from typing import Tuple
import torch
import numpy as np

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    i= torch.arange(0, head_dim, 2)[:(head_dim // 2)]
    theta_i =  (theta ** (-i / head_dim))
    m = torch.arange(seqlen).float()
    frequencies = torch.outer(m, theta_i)
    #frequencies = torch.from_numpy(frequencies).float()
    cos = torch.cos(frequencies)
    sin = torch.sin(frequencies)
    cos = reshape_for_broadcast(cos, query_real)
    sin = reshape_for_broadcast(sin, query_imag)    
    query_real_rot = cos * query_real - sin * query_imag
    query_imag_rot = sin * query_real + cos * query_imag
    key_real_rot = cos * key_real - sin * key_imag
    key_imag_rot = sin * key_real + cos * key_imag
    
    query_out = torch.stack([query_real_rot, query_imag_rot], dim=-1)
    query_out = query_out.reshape(query.shape)
    key_out = torch.stack((key_real_rot, key_imag_rot), dim= -1)
    key_out = key_out.reshape(key.shape)

    '''  query_out = torch.from_numpy(query_out)
    key_out = torch.from_numpy(key_out)'''
    
    return query_out, key_out