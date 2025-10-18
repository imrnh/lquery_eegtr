import torch
from model.transformer import Transformer
from config import config, device



# Create a dummy input batch
batch_size, timestamp, num_channels, embed_dim = 1, 99, 26, config['model_embed_dim']
eeg_signal = torch.rand((batch_size, timestamp, num_channels, embed_dim)).to(device) # (b, timestamps, num_channels, embed_dim)


model = Transformer(**config).to(device)
model.train() # Set model to training mode to test dropout

print(f"Model created on {device} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

outputs = model(eeg_signal)

print(outputs.shape)