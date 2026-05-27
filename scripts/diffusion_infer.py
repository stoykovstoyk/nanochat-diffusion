"""
Diffusion LLM Inference and Generation Script.

Usage:
    python -m scripts.diffusion_infer
    python -m scripts.diffusion_infer --prompt "Hello world" --model-name diffusion

Adapted from karpathy/nanochat for diffusion LLM inference.
"""

import os
import argparse
import json
import time

import torch
import torch.nn.functional as F

from nanochat_diffusion.gpt import GPT, GPTConfig, Linear
from nanochat_diffusion.diffusion_model import DiffusionModel, DiffusionConfig
from nanochat_diffusion.diffusion_sampler import DiffusionSampler
from nanochat_diffusion.tokenizer import Tokenizer, UNK_TOKEN_ID
from nanochat_diffusion.common import print0, print_banner, get_base_dir, compute_init, autodetect_device_type
from nanochat_diffusion.checkpoint_manager import load_model

print_banner()


def setup_model(model_name="diffusion", checkpoint_dir="", device="cuda", **kwargs):
    """Setup and load a diffusion or GPT model."""
    if model_name == "diffusion":
        print0("Loading Diffusion LLM...")
        model, tokenizer = load_model(
            model_name=model_name,
            device=device,
            phase="eval",
            checkpoint_dir=checkpoint_dir,
            **kwargs
        )
        model.eval()
    elif model_name == "gpt":
        print0("Loading GPT model...")
        model, tokenizer = load_model(
            model_name=model_name,
            device=device,
            phase="eval",
            checkpoint_dir=checkpoint_dir,
            **kwargs
        )
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    return model, tokenizer


def diffusion_sampling(model, prompt, max_tokens=128, temperature=1.0, top_k=40, num_steps=20,
                       tokenizer=None):
    """Generate text using diffusion sampling from a prompt."""
    from nanochat_diffusion.tokenizer import UNK_TOKEN_ID
    
    device = model.get_device()
    config = model.config
    unk_id = config.unk_token_id
    
    # Encode prompt to tokens
    if isinstance(prompt, str):
        if tokenizer is not None:
            prompt_tokens = tokenizer.encode(prompt, prepend=False)
        else:
            prompt_tokens = [b for b in prompt.encode('utf-8')]
    else:
        prompt_tokens = list(prompt)
    
    # Start with all UNK tokens
    seq_len = max_tokens + len(prompt_tokens)
    current_tokens = torch.full(
        (1, seq_len),
        unk_id,
        dtype=torch.long,
        device=device
    )
    
    # Fill in the prompt at the beginning
    for i, tok in enumerate(prompt_tokens):
        current_tokens[0, i] = tok
    
    print0(f"Starting diffusion sampling from UNK (seq_len={seq_len}, steps={num_steps})")
    
    # Progressive denoising
    history = []
    for step in range(num_steps):
        print0(f"Step {step + 1}/{num_steps}", end="\r")
        
        # Forward pass to get logits for UNK positions
        with torch.no_grad():
            logits = model(current_tokens)
        
        # Get logits for UNK positions
        unk_mask = (current_tokens[0] == unk_id).bool()
        unk_positions = torch.where(unk_mask)[0]
        
        if len(unk_positions) == 0:
            print0("\nAll tokens determined!")
            break
        
        # Sample for each UNK position
        for pos in unk_positions:
            logit = logits[0, pos]
            if temperature > 0:
                logit = logit / temperature
                if top_k > 0:
                    top_k_values, _ = torch.topk(logit, min(top_k, logit.size(-1)))
                    min_val = top_k_values[-1]
                    mask = logit < min_val
                    logit = logit.masked_fill(mask, -1e10)
                probs = F.softmax(logit, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(logit).item()
        
            current_tokens[0, pos] = next_token
        
        history.append(current_tokens[0].clone())
    
    print()
    return current_tokens[0], history


def autoregressive_generate(model, prompt, max_tokens=128, temperature=1.0, top_k=40,
                            tokenizer=None):
    """Generate text using standard autoregressive approach."""
    from nanochat_diffusion.tokenizer import UNK_TOKEN_ID
    
    device = model.get_device()
    config = model.config
    
    if isinstance(prompt, str):
        if tokenizer is not None:
            prompt_tokens = tokenizer.encode(prompt, prepend=False)
        else:
            prompt_tokens = [b for b in prompt.encode('utf-8')]
    else:
        prompt_tokens = list(prompt)
    
    # Start with prompt, fill rest with UNK
    seq_len = max_tokens + len(prompt_tokens)
    current_tokens = torch.full(
        (1, seq_len),
        config.unk_token_id,
        dtype=torch.long,
        device=device
    )
    
    for i, tok in enumerate(prompt_tokens):
        current_tokens[0, i] = tok
    
    # Sample each position
    generated = []
    for pos in range(len(prompt_tokens), seq_len):
        with torch.no_grad():
            logits = model(current_tokens)
        
        logit = logits[0, pos]
        if temperature > 0:
            logit = logit / temperature
            if top_k > 0:
                top_k_values, _ = torch.topk(logit, min(top_k, logit.size(-1)))
                min_val = top_k_values[-1]
                mask = logit < min_val
                logit = logit.masked_fill(mask, -1e10)
            probs = F.softmax(logit, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        else:
            next_token = torch.argmax(logit).item()
        
        current_tokens[0, pos] = next_token
        generated.append(next_token)
    
    return generated, current_tokens[0].tolist()


def evaluate_diffusion(model, test_data, metrics=None):
    """Evaluate the diffusion model on test data."""
    from nanochat_diffusion.tokenizer import UNK_TOKEN_ID
    
    device = model.get_device()
    config = model.config
    unk_id = config.unk_token_id
    
    if metrics is None:
        metrics = {'loss': [], 'accuracy': []}
    
    model.eval()
    with torch.no_grad():
        for batch in test_data:
            # Forward pass
            logits = model(batch)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.config.vocab_size),
                batch.view(-1),
                ignore_index=-1
            )
            metrics['loss'].append(loss.item())
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == batch).float().mean()
            metrics['accuracy'].append(acc.item())
    
    return {k: sum(v) / len(v) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Diffusion LLM Inference")
    
    # Model config
    parser.add_argument("--model", type=str, default="diffusion", help="model type")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="checkpoint directory")
    parser.add_argument("--checkpoint-step", type=str, default="latest", help="checkpoint step")
    parser.add_argument("--device", type=str, default="", help="device (auto=autodetect)")
    
    # Sampling config
    parser.add_argument("--seq-len", type=int, default=256, help="sequence length")
    parser.add_argument("--max-tokens", type=int, default=128, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="top-k filtering")
    parser.add_argument("--num-steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--prompt", type=str, default="", help="text prompt to continue")
    
    # Inference mode
    parser.add_argument("--mode", type=str, default="diffusion", 
                       choices=["diffusion", "autoregressive", "both"],
                       help="inference mode")
    parser.add_argument("--output-file", type=str, default="", help="save output to file")
    
    args = parser.parse_args()
    
    # Setup device
    device_type = args.device or autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    
    # Load model
    model, tokenizer = setup_model(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        device=device
    )
    
    print0(f"Device: {device}")
    print0(f"Model: {args.model}")
    print0(f"Checkpoint: {args.checkpoint_step}")
    
    if args.prompt:
        print0(f"Prompt: {args.prompt}")
        
        if args.mode in ["diffusion", "both"]:
            print0("\n" + "="*80)
            print0("Diffusion Sampling Mode")
            print0("="*80)
            generated, history = diffusion_sampling(
                model, args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                num_steps=args.num_steps,
                tokenizer=tokenizer
            )
            
            decoded = tokenizer.decode(generated.tolist())
            print(f"\nGenerated text:\n{decoded}")
        
        if args.mode in ["autoregressive", "both"]:
            print0("\n" + "="*80)
            print0("Autoregressive Generation Mode")
            print0("="*80)
            generated, history = autoregressive_generate(
                model, args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                tokenizer=tokenizer
            )
            
            decoded = tokenizer.decode(generated)
            print(f"\nGenerated text:\n{decoded}")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"Prompt: {args.prompt}\n\nGenerated:\n")
                f.write(decoded)
            print0(f"\nOutput saved to {args.output_file}")
    
    else:
        # Run quick evaluation without prompt
        print0("\n" + "="*80)
        print0("Running quick evaluation")
        print0("="*80)
        
        # Generate random samples
        for i in range(3):
            print0(f"\nSample {i+1}:")
            generated, _ = diffusion_sampling(
                model, "Hello",
                max_tokens=64,
                temperature=0.7,
                top_k=30,
                num_steps=15
            )
            decoded = tokenizer.decode(generated.tolist())
            print(f"Generated: {decoded}")
    
    print0("\n" + "="*80)
    print0("Inference complete!")
    print0("="*80)


if __name__ == "__main__":
    main()
