import os
import time
import math
import pickle
import numpy as np
import torch
from torch.nn import functional as F
from contextlib import nullcontext
from model import GPTConfig, GPT
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Optimal hyperparameters found by Optuna
opt_config = {
    "n_layer": 9,
    "n_embd": 768,
    "learning_rate": 0.0005212377267634382,
    "dropout": 0.15935224572057916,
    "weight_decay": 0.008591969753189597
}

config = {
    "out_dir": "out-tiny-stories-final",
    "eval_interval": 250,
    "eval_iters": 200,
    "log_interval": 100,
    "eval_only": False,
    "always_save_checkpoint": True,
    "init_from": "scratch",
    "wandb_log": False,  # Disabled WandB logging
    "dataset": "tiny_stories",
    "gradient_accumulation_steps": 2,
    "batch_size": 40,
    "block_size": 256,
    "bias": False,
    "n_layer": opt_config["n_layer"],
    "n_head": 8,
    "n_embd": opt_config["n_embd"],
    "dropout": opt_config["dropout"],
    "grad_clip": 1.0,
    "beta1": 0.9,
    "weight_decay": opt_config["weight_decay"],
    "learning_rate": opt_config["learning_rate"],
    "max_iters": 100000,
    "lr_decay_iters": 100000,
    "min_lr": 1e-6,
    "beta2": 0.98,
    "decay_lr": True,
    "warmup_iters": 1000,
    "device": "cuda",
    "dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
    "compile": True,
}

# Create output directory if it doesn't exist
os.makedirs(config['out_dir'], exist_ok=True)

def get_batch(data_dir, split, block_size, batch_size, device):
    assert split in ['train', 'val'], f"Invalid split: {split}"

    data_file = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(data_file, dtype=np.uint16, mode='r')

    indices = torch.randint(len(data) - block_size, (batch_size,))
    input_data = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in indices])
    target_data = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in indices])

    if device.type == 'cuda':
        input_data, target_data = input_data.pin_memory().to(device, non_blocking=True), target_data.pin_memory().to(device, non_blocking=True)
    else:
        input_data, target_data = input_data.to(device), target_data.to(device)

    return input_data, target_data

@torch.no_grad()
def estimate_loss(model, data_dir, eval_iters, block_size, batch_size, device, ctx):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            input_data, target_data = get_batch(data_dir, split, block_size, batch_size, device)
            with ctx:
                _, loss = model(input_data, target_data)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean()
    model.train()
    return losses

def get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

device = torch.device(config['device'])
dtype = getattr(torch, config['dtype'])
ctx = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type=device.type, dtype=dtype)

# Initialize the model
model_args = {
    'n_layer': config['n_layer'],
    'n_head': config['n_head'],
    'n_embd': config['n_embd'],
    'block_size': config['block_size'],
    'bias': config['bias'],
    'vocab_size': None,
    'dropout': config['dropout'],
}

data_dir = os.path.join('data', config['dataset'])
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    model_args['vocab_size'] = meta['vocab_size']
    print(f"Found vocab_size = {model_args['vocab_size']} (inside {meta_path})")
else:
    model_args['vocab_size'] = 50304
    print(f"Defaulting to vocab_size of {model_args['vocab_size']}")

gptconf = GPTConfig(**model_args)
model = GPT(gptconf).to(device)

# Initialize optimizer and GradScaler
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
scheduler = CosineAnnealingLR(optimizer, T_max=config['lr_decay_iters'], eta_min=config['min_lr'])
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))

# Compile the model
if config['compile']:
    print("Compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

iter_num = 0
best_val_loss = float('inf')
start_time = time.time()

while iter_num < config['max_iters']:
    lr = get_lr(iter_num, config['learning_rate'], config['warmup_iters'], config['lr_decay_iters'], config['min_lr']) if config['decay_lr'] else config['learning_rate']
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % config['eval_interval'] == 0:
        losses = estimate_loss(
            model, data_dir, config['eval_iters'], config['block_size'], config['batch_size'], device, ctx
        )
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or config['always_save_checkpoint']:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to {config['out_dir']}")
                torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))

    input_data, target_data = get_batch(data_dir, 'train', config['block_size'], config['batch_size'], device)
    for _ in range(config['gradient_accumulation_steps']):
        with ctx:
            _, loss = model(input_data, target_data)
            loss = loss / config['gradient_accumulation_steps']
        scaler.scale(loss).backward()
    if config['grad_clip'] != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    if iter_num % config['log_interval'] == 0:
        elapsed_time = time.time() - start_time
        print(f"Iter {iter_num}: loss {loss.item():.4f}, time {elapsed_time * 1000:.2f}ms")
    iter_num += 1
    start_time = time.time()

print("Training complete. Best validation loss: ", best_val_loss)
