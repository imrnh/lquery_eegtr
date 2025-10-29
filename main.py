import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler 
from types import SimpleNamespace

from train import train
from convolution import FrequencyBinConvolution
from model.transformer import Transformer
from data_processor import EEGDataset
from config import config, device, train_dir, val_dir

# Defining Hyperparameters
hyperparams = SimpleNamespace(
    # Dataloader parameters
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,

    # Training Hyperparameters
    learning_rate=1e-3,
    num_epochs=10,
    
    # Training checkpointing and evaluation
    step=0,  # Global step counter
    training_state_checkpoint_frequency=1000,
    eval_interval=500, 
)


# Data processor
train_ds = EEGDataset(train_dir, metadata_file="lib/train_metadata.pkl")
val_ds = EEGDataset(val_dir, metadata_file="lib/val_metadata.pkl")

dataloader = DataLoader(train_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_dataloader = DataLoader(val_ds, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)


#  Frequency bin convolution and Transformer Model
freq_bin_conv = FrequencyBinConvolution(embed_dim=config['model_embed_dim']).to(device)
model = Transformer(**config).to(device)
model.train() # Set model to training mode to test dropout


# Optimizer and Loss
optimizer = torch.optim.AdamW(list(model.parameters()) + list(freq_bin_conv.parameters()),lr = hyperparams.learning_rate)
kl_loss = nn.KLDivLoss(reduction='batchmean')  # requires log_probs input


# Start training
scaler = GradScaler(device=device)
train(
    model=model,
    freq_bin_conv=freq_bin_conv,
    dataloader=dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    scaler=scaler,
    kl_loss=kl_loss,
    device=device,
    config=hyperparams,
    start_epoch=0,
    checkpoint_path="checkpoint.pt",
    best_models_dir="model_checkpoints"
)
