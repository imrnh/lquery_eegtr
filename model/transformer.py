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
    def __init__(self, output_dim, model_embed_dim, num_layers, num_heads, d_ff,
                  num_learnable_queries,
                 dropout_attn_xp, dropout_attn_lqp, dropout_mlp, dropout_embed):
        super().__init__()
        self.model_embed_dim = model_embed_dim
        self.num_learnable_queries = num_learnable_queries

        # Learnable Query Tokens 
        if self.num_learnable_queries > 0:
            self.learnable_queries = nn.Parameter(
                torch.randn(1, self.num_learnable_queries, model_embed_dim)
            )

        # Encoder Stack
        self.layers = nn.ModuleList([
            EncoderBlock(
                model_embed_dim=model_embed_dim,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_attn=0.0, # Placeholder, not used in custom MHA
                dropout_attn_xp=dropout_attn_xp,
                dropout_attn_lqp=dropout_attn_lqp,
                dropout_mlp=dropout_mlp
            ) for _ in range(num_layers)
        ])

        # Output Head
        self.final_norm = nn.LayerNorm(model_embed_dim)
        self.lm_head = nn.Linear(model_embed_dim, output_dim)

    def forward(self, x):
        b, timestamps, num_channels, embed_dim = x.shape

        # pass input to the encoder. No additional processing.
        # lq = torch.rand((bsz, (num_channels), embed_dim))
       
        for lix, layer in enumerate(self.layers):
            num_x_tokens = x.shape[1] # number of original tokens before adding learnable queries.

            # Add learnable queries if layer index > 0
            if lix > 0 and self.num_learnable_queries > 0:
                lq_expanded = self.learnable_queries.unsqueeze(1).expand(b, timestamps, -1, -1)  # (bsz, timestamps, self.num_learnable_queries, model_embed_dim) 

                print(f"Shape of X: {x.shape}")
                print(f"Shape of LQ: {lq_expanded.shape}")
                
                x = torch.cat([x, lq_expanded], dim=2)

                print(f"Modified x shape: {x.shape}")

            # Pass to encoder.
            x = layer(x, num_x_tokens=num_x_tokens)


        processed_output = self.final_norm(x)
        embeddings = self.lm_head(processed_output)

        return embeddings