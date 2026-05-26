"""
Diffusion LLM Evaluation Script.

Usage:
    python -m scripts.diffusion_evaluate --model diffusion --checkpoint-step latest
    python -m scripts.diffusion_evaluate --model diffusion --tasks gsm8k,arc

Adapted from karpathy/nanochat.
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
from nanochat_diffusion.diffusion_scheduler import (
    create_noise_schedule,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
)

print_banner()


def evaluate_perplexity(model, test_data, tokenizer, device, seq_len=256):
    """Evaluate perplexity on test data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data):
            # Mask some tokens for diffusion evaluation
            if isinstance(model, DiffusionModel):
                B, T = batch.shape
                t = torch.zeros(B, device=device, dtype=torch.long)
                masked = model.mask_tokens(batch, t)
                logits = model(masked, t=t)
            else:
                logits = model(batch)
            
            # Compute cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.view(-1),
                ignore_index=-1
            )
            
            total_loss += loss.item()
            total_tokens += batch.numel()
    
    avg_loss = total_loss / max(1, len(test_data))
    perplexity = torch.exp(torch.tensor(avg_loss))
    return {
        'perplexity': float(perplexity),
        'avg_loss': avg_loss,
        'total_tokens': total_tokens,
    }


def evaluate_generation_quality(model, prompt, max_tokens=128, temperature=1.0, num_steps=20):
    """Evaluate generation quality with sampling."""
    device = model.get_device()
    config = model.config
    
    if isinstance(prompt, str):
        prompt_tokens = [ord(c) for c in prompt]
    else:
        prompt_tokens = list(prompt)
    
    # Initialize with all UNK
    seq_len = max_tokens + len(prompt_tokens)
    current_tokens = torch.full(
        (1, seq_len),
        config.unk_token_id,
        dtype=torch.long,
        device=device
    )
    
    # Fill in the prompt
    for i, tok in enumerate(prompt_tokens):
        current_tokens[0, i] = tok
    
    # Progressive denoising
    history = []
    for step in range(num_steps):
        with torch.no_grad():
            logits = model(current_tokens)
        
        unk_mask = (current_tokens[0] == config.unk_token_id).bool()
        unk_positions = torch.where(unk_mask)[0]
        
        if len(unk_positions) == 0:
            break
        
        for pos in unk_positions:
            logit = logits[0, pos]
            if temperature > 0:
                logit = logit / temperature
                probs = F.softmax(logit, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(logit).item()
            
            current_tokens[0, pos] = next_token
        
        history.append(current_tokens[0].clone())
    
    return {
        'final_tokens': current_tokens[0].tolist(),
        'history': [h.tolist() for h in history],
        'steps_taken': len(history),
    }


def evaluate_consistency(model, test_data, num_samples=10, num_steps=20):
    """Evaluate generation consistency across multiple runs."""
    model.eval()
    samples = []
    
    for i in range(num_samples):
        # Random UNK mask
        seq_len = 128
        current_tokens = torch.full(
            (1, seq_len),
            model.config.unk_token_id,
            dtype=torch.long,
            device=model.get_device()
        )
        
        # Denoise
        for step in range(num_steps):
            with torch.no_grad():
                logits = model(current_tokens)
            
            unk_mask = (current_tokens[0] == model.config.unk_token_id).bool()
            if unk_mask.sum() == 0:
                break
            
            for pos in torch.where(unk_mask)[0]:
                logit = logits[0, pos]
                probs = F.softmax(logit / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                current_tokens[0, pos] = next_token
        
        samples.append(current_tokens[0].tolist())
    
    # Compute consistency metrics
    # This is a simplified version - in practice you'd want better metrics
    return {
        'num_samples': num_samples,
        'samples': samples,
        'mean_seq_len': sum(len(s) for s in samples) / len(samples),
    }


def evaluate_noise_schedules(model, test_data, schedules=None):
    """Evaluate different noise schedules."""
    if schedules is None:
        schedules = {
            'linear': LinearNoiseSchedule,
            'cosine': CosineNoiseSchedule,
        }
    
    results = {}
    for name, schedule_cls in schedules.items():
        schedule = schedule_cls(num_steps=1000, max_mask=0.8)
        # Test with the schedule
        result = evaluate_perplexity(model, test_data, schedule)
        results[name] = result
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Diffusion LLM Evaluation")
    
    # Model config
    parser.add_argument("--model", type=str, default="diffusion", help="model type")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="checkpoint directory")
    parser.add_argument("--checkpoint-step", type=str, default="latest", help="checkpoint step")
    parser.add_argument("--device", type=str, default="", help="device")
    
    # Evaluation config
    parser.add_argument("--seq-len", type=int, default=256, help="sequence length")
    parser.add_argument("--max-tokens", type=int, default=128, help="max tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature")
    parser.add_argument("--num-steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--num-samples", type=int, default=10, help="num samples for consistency")
    
    # Tasks
    parser.add_argument("--tasks", type=str, default="gsm8k,arc", help="tasks to evaluate")
    parser.add_argument("--output-file", type=str, default="evaluation_results.json", help="output file")
    
    # Evaluation modes
    parser.add_argument("--evaluate-perplexity", action="store_true", help="evaluate perplexity")
    parser.add_argument("--evaluate-generation", action="store_true", help="evaluate generation quality")
    parser.add_argument("--evaluate-consistency", action="store_true", help="evaluate consistency")
    parser.add_argument("--evaluate-noise-schedules", action="store_true", help="evaluate noise schedules")
    
    args = parser.parse_args()
    
    # Setup device
    device_type = args.device or autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    
    # Load model
    model, tokenizer = load_model(
        model_name=args.model,
        device=device,
        phase="eval",
        checkpoint_dir=args.checkpoint_dir,
    )
    
    print0(f"Device: {device}")
    print0(f"Model: {args.model}")
    print0(f"Checkpoint: {args.checkpoint_step}")
    
    # Evaluate perplexity
    if args.evaluate_perplexity:
        print0("\nEvaluating perplexity...")
        test_data = [
            torch.randint(0, 32768, (1, args.seq_len), device=device)
            for _ in range(10)
        ]
        metrics = evaluate_perplexity(model, test_data, tokenizer, device, args.seq_len)
        print0(f"Perplexity: {metrics['perplexity']:.4f}")
        print0(f"Avg loss: {metrics['avg_loss']:.4f}")
        print0(f"Total tokens: {metrics['total_tokens']}")
    
    # Evaluate generation
    if args.evaluate_generation:
        print0("\nEvaluating generation quality...")
        result = evaluate_generation_quality(
            model, "Hello world",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_steps=args.num_steps
        )
        print0(f"Generation took {result['steps_taken']} steps")
        print(f"Generated: {result['final_tokens'][:20]}")
    
    # Evaluate consistency
    if args.evaluate_consistency:
        print0("\nEvaluating consistency...")
        metrics = evaluate_consistency(model, [], args.num_samples, args.num_steps)
        print0(f"Evaluated {metrics['num_samples']} samples")
    
    # Evaluate noise schedules
    if args.evaluate_noise_schedules:
        print0("\nEvaluating noise schedules...")
        schedules = evaluate_noise_schedules(model, [], [
            LinearNoiseSchedule,
            CosineNoiseSchedule,
        ])
        for name, metrics in schedules.items():
            print0(f"{name}: {metrics}")
    
    print0("\nEvaluation complete!")


if __name__ == "__main__":
    main()
