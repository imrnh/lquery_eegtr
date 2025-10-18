import torch
from model.transformer import Transformer

config = {
    'vocab_size': 1000,
    'd_model': 512,
    'num_layers': 5,
    'num_heads': 16,
    'd_ff': 2048, # 4 * d_model
    'context_length': 256,
    'num_learnable_queries': 16, # Number of special query tokens
    'dropout_attn_xp': 0.3,  # Dropout for standard self-attention
    'dropout_attn_lqp': 0.1, # Higher dropout for attention involving queries
    'dropout_mlp': 0.1,
    'dropout_embed': 0.1,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create the model instance
model = Transformer(**config).to(device)
print(f"Model created on {device} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

# Create a dummy input batch
batch_size = 12
seq_len = 26
dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)

eeg_signal = torch.rand((100, 100))


# Perform a forward pass
model.train() # Set model to training mode to test dropout
logits = model(dummy_input)

# Print output shape
print(f"Input shape: {dummy_input.shape}")
print(f"Output (logits) shape: {logits.shape}")
# Expected output shape: (batch_size, seq_len, vocab_size)
assert logits.shape == (batch_size, seq_len, config['vocab_size'])

print("\nForward pass successful!")
