import torch
import torch.nn as nn
from model.encoder_block import EncoderBlock

class Transformer(nn.Module):
    """
    A complete Transformer model using the custom EncoderBlock.

    This model includes token and positional embeddings, a stack of encoder blocks,
    and a final head for projecting to the vocabulary size, making it suitable
    for language modeling tasks.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff,
                 context_length, num_learnable_queries,
                 dropout_attn_xp, dropout_attn_lqp, dropout_mlp, dropout_embed):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_learnable_queries = num_learnable_queries

        # --- Embedding Layers ---
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_length, d_model)
        self.embed_dropout = nn.Dropout(dropout_embed)

        # --- Learnable Query Tokens (optional) ---
        if self.num_learnable_queries > 0:
            self.learnable_queries = nn.Parameter(
                torch.randn(1, self.num_learnable_queries, d_model)
            )

        # --- Encoder Stack ---
        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_attn=0.0, # Placeholder, not used in custom MHA
                dropout_attn_xp=dropout_attn_xp,
                dropout_attn_lqp=dropout_attn_lqp,
                dropout_mlp=dropout_mlp
            ) for _ in range(num_layers)
        ])

        # --- Output Head ---
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Tie the weights of the token embedding and the final linear layer
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx):
        """
        Forward pass of the Transformer model.

        Args:
            idx (torch.Tensor): Input tensor of token indices, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits over the vocabulary, shape (batch_size, seq_len, vocab_size).
        """
        b, t = idx.shape
        assert t <= self.context_length, \
            f"Input sequence length ({t}) exceeds model's context length ({self.context_length})"
            
        # --- 1. Get Embeddings ---
        tok_embed = self.token_embedding(idx)
        pos_ids = torch.arange(t, device=idx.device)
        pos_embed = self.pos_embedding(pos_ids)
        x = self.embed_dropout(tok_embed + pos_embed)
        num_x_tokens = x.shape[1]

        # --- 2. Concatenate with Learnable Queries (if any) ---
        if self.num_learnable_queries > 0:
            # Expand queries to match the batch size
            lq = self.learnable_queries.expand(b, -1, -1)
            x = torch.cat([x, lq], dim=1)

        # --- 3. Pass through Encoder Stack ---
        for layer in self.layers:
            x = layer(x, num_x_tokens=num_x_tokens)

        # --- 4. Process Output ---
        # Normalize the full output
        processed_output = self.final_norm(x)

        # For language modeling, we only care about the outputs corresponding
        # to the original tokens, not the learnable queries.
        processed_tokens = processed_output[:, :num_x_tokens, :]

        # --- 5. Get Logits ---
        logits = self.lm_head(processed_tokens)

        return logits