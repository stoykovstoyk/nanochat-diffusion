"""
Checkpoint management for diffusion LLM.
Adapted from karpathy/nanochat.
"""

import os
import json
import time
import torch
from nanochat_diffusion.common import get_base_dir, print0
from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig
from nanochat_diffusion.gpt import GPTConfig

def get_checkpoint_dir(model_name="base", phase="train"):
    base_dir = get_base_dir()
    phase_dir = os.path.join(base_dir, "checkpoints", model_name, phase)
    os.makedirs(phase_dir, exist_ok=True)
    return phase_dir

def save_checkpoint(model, optimizer, step, loss, metrics, model_name="base", phase="train", 
                   include_optimizer=True, include_scheduler=True):
    """Save model checkpoint with metadata"""
    if isinstance(model, DiffusionModel):
        config = model.config
    else:
        config = GPTConfig()
    
    checkpoint_dir = get_checkpoint_dir(model_name, phase)
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step:010d}")
    
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model weights
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))
    
    # Save config
    if isinstance(config, DiffusionConfig):
        config_dict = {
            'model_type': 'diffusion',
            'diffusion_config': {
                'sequence_len': config.sequence_len,
                'vocab_size': config.vocab_size,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_kv_head': config.n_kv_head,
                'n_embd': config.n_embd,
                'num_diffusion_steps': config.num_diffusion_steps,
                'unk_token_id': config.unk_token_id,
                'max_mask_ratio': config.max_mask_ratio,
            },
            'gpt_config': {
                'sequence_len': config.sequence_len,
                'vocab_size': config.vocab_size,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_kv_head': config.n_kv_head,
                'n_embd': config.n_embd,
                'window_pattern': config.window_pattern,
            }
        }
    else:
        config_dict = {
            'model_type': 'gpt',
            'gpt_config': {
                'sequence_len': config.sequence_len,
                'vocab_size': config.vocab_size,
                'n_layer': config.n_layer,
                'n_head': config.n_head,
                'n_kv_head': config.n_kv_head,
                'n_embd': config.n_embd,
                'window_pattern': getattr(config, 'window_pattern', 'SSSL'),
            }
        }
    
    with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Save optimizer state
    if include_optimizer and optimizer is not None:
        if hasattr(optimizer, 'state_dict'):
            torch.save(optimizer.state_dict(), 
                      os.path.join(checkpoint_path, "optimizer.pt"))
    
    # Save training metadata
    metadata = {
        'step': step,
        'loss': float(loss),
        'metrics': metrics or {},
        'timestamp': time.time(),
    }
    with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print0(f"Saved checkpoint at step {step} to {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(model, optimizer=None, step="latest", model_name="base", phase="train", 
                   load_optimizer=True):
    """Load model checkpoint with metadata"""
    checkpoint_dir = get_checkpoint_dir(model_name, phase)
    
    if step == "latest":
        # Find the latest checkpoint
        try:
            checkpoints = sorted([int(d.split('_')[-1]) for d in os.listdir(checkpoint_dir) 
                                if d.startswith('step_')], reverse=True)
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            step_to_load = checkpoints[0]
        except (FileNotFoundError, ValueError):
            print0("No checkpoints found, starting from scratch")
            return None, None
    else:
        step_to_load = int(step)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step_to_load:010d}")
    model_path = os.path.join(checkpoint_path, "model.pt")
    
    if not os.path.exists(model_path):
        print0(f"No model file at {model_path}")
        return None, None
    
    # Load model
    state_dict = torch.load(model_path, map_location='cpu')
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    
    print0(f"Loaded model from step {step_to_load}")
    
    # Load optimizer if requested
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    if load_optimizer and optimizer is not None and os.path.exists(optimizer_path):
        optimizer_state = torch.load(optimizer_path, map_location='cpu')
        if hasattr(optimizer, 'load_state_dict'):
            optimizer.load_state_dict(optimizer_state)
        print0(f"Loaded optimizer from step {step_to_load}")
    
    # Load metadata
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print0(f"Checkpoint metadata: {metadata}")
        return model, metadata
    return model, None

def load_model(model_name="base", device="cuda", phase="eval", **kwargs):
    """Factory function to load model and optionally tokenizer"""
    from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig
    from nanochat_diffusion.tokenizer import Tokenizer
    
    # Create model
    config = DiffusionConfig(**kwargs)
    model = DiffusionModel(config)
    model.to(device)
    model.eval()
    
    # Load checkpoint if exists
    if kwargs.get('checkpoint_path'):
        load_checkpoint(model, model_name=model_name, phase=phase, step=kwargs.get('checkpoint_step', 'latest'))
    
    # Create tokenizer
    tokenizer = Tokenizer(data_dir="", verbose=False)
    
    return model, tokenizer
