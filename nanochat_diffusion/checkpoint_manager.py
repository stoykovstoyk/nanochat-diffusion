"""
Checkpoint management for diffusion LLM.
"""

import os
import json
import time
import torch
from nanochat_diffusion.common import get_base_dir, print0
from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig


def get_checkpoint_dir(model_name="diffusion", phase="train"):
    base_dir = get_base_dir()
    phase_dir = os.path.join(base_dir, "checkpoints", model_name, phase)
    os.makedirs(phase_dir, exist_ok=True)
    return phase_dir


def save_checkpoint(model, optimizer, step, loss, metrics, model_name="diffusion", phase="train",
                   include_optimizer=True):
    """Save model checkpoint with metadata"""
    # Unwrap torch.compile wrapper to access real model
    unwrapped = model
    if hasattr(model, '_orig_mod'):
        unwrapped = model._orig_mod
    elif hasattr(model, 'module'):
        unwrapped = model.module
    config = unwrapped.config if isinstance(unwrapped, DiffusionModel) else DiffusionConfig()

    checkpoint_dir = get_checkpoint_dir(model_name, phase)
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step:010d}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save model weights — unwrap torch.compile wrapper if present
    if hasattr(model, '_orig_mod'):
        state_dict = model._orig_mod.state_dict()
    elif hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(checkpoint_path, "model.pt"))

    # Save config
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

    with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    if include_optimizer and optimizer is not None:
        if hasattr(optimizer, 'state_dict'):
            torch.save(optimizer.state_dict(),
                      os.path.join(checkpoint_path, "optimizer.pt"))

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
    
    # Load model — strip _orig_mod. prefix for backward compat with compiled checkpoints
    state_dict = torch.load(model_path, map_location='cpu')
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
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
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print0(f"Checkpoint metadata: {metadata}")
            return model, metadata
        except (json.JSONDecodeError, UnicodeDecodeError):
            print0(f"Warning: corrupted metadata.json at {metadata_path}, ignoring")
            return model, None
    return model, None

def load_model(model_name="base", device="cuda", phase="eval", **kwargs):
    """Factory function to load model and optionally tokenizer.
    
    Loads config.json from the checkpoint to set model architecture,
    then loads model weights. Falls back to kwargs/defaults if no checkpoint.
    """
    from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig
    from nanochat_diffusion.tokenizer import Tokenizer
    from dataclasses import fields, asdict
    
    # Extract operational kwargs
    checkpoint_dir = kwargs.pop("checkpoint_dir", None) or kwargs.pop("checkpoint_path", None)
    checkpoint_step = kwargs.pop("checkpoint_step", "latest")
    
    # Determine checkpoint path
    if not checkpoint_dir:
        checkpoint_dir = get_checkpoint_dir(model_name=model_name, phase=phase)
    
    # Try to load config from checkpoint json
    config_dict = None
    resolved_step = checkpoint_step
    if os.path.exists(checkpoint_dir):
        if checkpoint_step == "latest":
            try:
                steps = sorted([int(d.split('_')[-1]) for d in os.listdir(checkpoint_dir)
                                if d.startswith('step_')], reverse=True)
                if steps:
                    resolved_step = str(steps[0])
            except (FileNotFoundError, ValueError):
                pass
        if resolved_step:
            try:
                step_int = int(resolved_step)
                config_path = os.path.join(checkpoint_dir, f"step_{step_int:010d}", "config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config_dict = json.load(f)
            except (ValueError, FileNotFoundError, json.JSONDecodeError):
                pass
    
    if config_dict and config_dict.get("diffusion_config"):
        # Build config from saved checkpoint metadata
        dc = config_dict["diffusion_config"]
        config = DiffusionConfig(
            sequence_len=dc.get("sequence_len", 1024),
            vocab_size=dc.get("vocab_size", 32768),
            n_layer=dc.get("n_layer", 12),
            n_head=dc.get("n_head", 6),
            n_kv_head=dc.get("n_kv_head", 6),
            n_embd=dc.get("n_embd", 768),
            window_pattern=dc.get("window_pattern", "SSSL"),
            num_diffusion_steps=dc.get("num_diffusion_steps", 1000),
            unk_token_id=dc.get("unk_token_id", 32767),
            max_mask_ratio=dc.get("max_mask_ratio", 0.8),
            sampling_steps=kwargs.pop("sampling_steps", 20),
            unmask_schedule=kwargs.pop("unmask_schedule", "linear"),
        )
    else:
        # Only pass valid DiffusionConfig fields from kwargs
        valid_fields = {f.name for f in fields(DiffusionConfig)}
        for key in list(kwargs):
            if key not in valid_fields:
                kwargs.pop(key, None)
        config = DiffusionConfig(**kwargs)
    
    # Create model
    model = DiffusionModel(config)
    model.to(device)
    model.eval()
    
    # Load checkpoint weights
    if os.path.exists(checkpoint_dir) and resolved_step:
        load_checkpoint(model, model_name=model_name, phase=phase, step=resolved_step)
    
    # Create tokenizer — use tokenizer saved in base_dir/tokenizer_diffusion/
    from nanochat_diffusion.common import get_base_dir
    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer_diffusion")
    tokenizer = Tokenizer(data_dir=tokenizer_dir, verbose=False)
    
    return model, tokenizer
