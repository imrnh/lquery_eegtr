import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    A Multi-Head Attention module that supports differential dropout for
    standard tokens and a separate set of learnable query tokens.

    This implementation is designed to be part of an encoder block that processes
    a single concatenated input tensor containing both standard tokens and
    learnable queries.
    """
    def __init__(self, d_in, d_out, dropout_xp, dropout_lqp, num_heads, qkv_bias=False):
        super().__init__()
        
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout_p_x = dropout_xp
        self.dropout_p_lq = dropout_lqp

    def forward(self, x, num_x_tokens):
        """
        Forward pass for Multi-Head Attention.

        Args:
            x (torch.Tensor): The combined input tensor of shape
                              (batch_size, num_tokens, embedding_dim).
                              It's assumed to be a concatenation of original
                              tokens and learnable queries.
            num_x_tokens (int): The number of original tokens in the input tensor.
                                This is used to distinguish them from the
                                learnable queries for applying differential dropout.
        """
        b, timestamps, num_tokens, d_in = x.shape
        num_lq_tokens = num_tokens - num_x_tokens

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape and transpose for multi-head processing
        keys = keys.view(b, timestamps,  num_tokens, self.num_heads, self.head_dim).transpose(2, 3)
        queries = queries.view(b, timestamps, num_tokens, self.num_heads, self.head_dim).transpose(2, 3)
        values = values.view(b, timestamps, num_tokens, self.num_heads, self.head_dim).transpose(2, 3)
        
        # --- Attention Calculation ---
        attn_scores = queries @ keys.transpose(-1, -2)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        # --- Apply Differential Dropout with Combined Scaling ---
        if self.training:
            keep_p_x = 1.0 - self.dropout_p_x
            keep_p_lq = 1.0 - self.dropout_p_lq
            
            if num_lq_tokens > 0:
                # Split the attention matrix into four blocks
                weights_xx = attn_weights[:, :, :num_x_tokens, :num_x_tokens]
                weights_xlq = attn_weights[:, :, :num_x_tokens, num_x_tokens:]
                weights_lqx = attn_weights[:, :, num_x_tokens:, :num_x_tokens]
                weights_lqlq = attn_weights[:, :, num_x_tokens:, num_x_tokens:]

                # Create binary masks for each block without scaling
                mask_xx = (torch.rand_like(weights_xx) < keep_p_x).float()
                mask_xlq = (torch.rand_like(weights_xlq) < keep_p_lq).float()
                mask_lqx = (torch.rand_like(weights_lqx) < keep_p_lq).float()
                mask_lqlq = (torch.rand_like(weights_lqlq) < keep_p_lq).float()

                # Apply the masks to get unscaled dropped-out weights
                masked_xx = weights_xx * mask_xx
                masked_xlq = weights_xlq * mask_xlq
                masked_lqx = weights_lqx * mask_lqx
                masked_lqlq = weights_lqlq * mask_lqlq
                
                # Recombine the masked matrix
                top_row = torch.cat([masked_xx, masked_xlq], dim=-1)
                bottom_row = torch.cat([masked_lqx, masked_lqlq], dim=-1)
                masked_attn_weights = torch.cat([top_row, bottom_row], dim=-2)
                
                # Calculate the single combined scaling factor
                n_xx = num_x_tokens * num_x_tokens
                n_lq_related = num_tokens**2 - n_xx
                n_total = num_tokens**2
                
                expected_total_kept = (n_xx * keep_p_x) + (n_lq_related * keep_p_lq)
                
                scale = n_total / expected_total_kept if expected_total_kept > 0 else 0.0
                
                # Apply the single scaling factor to the entire matrix
                attn_weights = masked_attn_weights * scale

            else: # Standard dropout if no learnable queries exist
                attn_weights = F.dropout(attn_weights, p=self.dropout_p_x, training=self.training)
        
        context_vec = (attn_weights @ values).transpose(2, 3)
        context_vec = context_vec.contiguous().view(b, timestamps, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec