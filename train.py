import os
import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm.auto import tqdm

from utils import get_vote_probability, load_validation_history, initialize_best_models_heap, log_validation_loss, save_checkpoint, maintain_best_models



def train_one_step(batch, model, freq_bin_conv, optimizer, scaler, kl_loss, device):
    """Execute one training step."""
    spectrograms, labels, votes = batch
    
    spectrograms = spectrograms.to(device, dtype=torch.float32, non_blocking=True)
    votes = get_vote_probability(votes).to(device, dtype=torch.float32, non_blocking=True)
    
    optimizer.zero_grad(set_to_none=True)
    
    with autocast("cuda", dtype=torch.float16):
        x = freq_bin_conv(spectrograms)
        out = model(x)
        log_probs = nn.functional.log_softmax(out, dim=-1)
        loss = kl_loss(log_probs, votes)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


def evaluate(model, freq_bin_conv, val_dataloader, kl_loss, device):
    """Run validation and return average loss."""
    model.eval()
    freq_bin_conv.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_spectrograms, val_labels, val_votes = val_batch
            val_spectrograms = val_spectrograms.to(device, dtype=torch.float32, non_blocking=True)
            val_votes = get_vote_probability(val_votes).to(device, dtype=torch.float32, non_blocking=True)
            
            with autocast(device_type="cuda", dtype=torch.float16):
                x = freq_bin_conv(val_spectrograms)
                val_out = model(x)
                val_log_probs = nn.functional.log_softmax(val_out, dim=-1)
                val_loss += kl_loss(val_log_probs, val_votes).item()
    
    model.train()
    freq_bin_conv.train()
    
    return val_loss / len(val_dataloader)


def train(
    model,
    freq_bin_conv,
    dataloader,
    val_dataloader,
    optimizer,
    scaler,
    kl_loss,
    device,
    config,
    start_epoch=0,
    checkpoint_path="checkpoint.pt",
    best_models_dir="model_checkpoints"
):
    """Main training loop."""
    # Setup directories and logging
    os.makedirs(best_models_dir, exist_ok=True)
    val_log_file = os.path.join(best_models_dir, "val_loss_log.csv")
    
    # Load validation history and initialize best models heap
    val_history = load_validation_history(val_log_file)
    best_models = initialize_best_models_heap(val_history, k=5)
    
    # CUDA timing events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        freq_bin_conv.train()
        epoch_loss = 0.0
        
        t = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in t:
            config.step += 1
            
            # Training step
            loss = train_one_step(batch, model, freq_bin_conv, optimizer, scaler, kl_loss, device)
            epoch_loss += loss
            t.set_postfix({'KL_loss': loss})
            
            # Save checkpoint periodically
            if config.step % config.training_state_checkpoint_frequency == 0:
                save_checkpoint(checkpoint_path, epoch, config.step, model, freq_bin_conv, optimizer, scaler)
                print(f"\n[Step {config.step}] Checkpoint saved to {checkpoint_path}")
            
            # Evaluation
            if config.step % config.eval_interval == 0:
                val_loss = evaluate(model, freq_bin_conv, val_dataloader, kl_loss, device)
                print(f"\n[Step {config.step}] Validation loss: {val_loss:.4f}")
                
                # Save validation checkpoint
                save_path = os.path.join(best_models_dir, f"model_step{config.step}_valloss{val_loss:.4f}.pt")
                save_checkpoint(save_path, epoch, config.step, model, freq_bin_conv, optimizer, scaler, val_loss)
                
                # Log and maintain best models
                log_validation_loss(val_log_file, config.step, val_loss, save_path)
                maintain_best_models(best_models, val_loss, save_path, max_models=5)
        
        print(f"Epoch {epoch+1} average loss: {epoch_loss/len(dataloader):.4f}")
    
    end.record()
    torch.cuda.synchronize()
    print(f"\nTraining completed in {start.elapsed_time(end)/1000:.2f} seconds")