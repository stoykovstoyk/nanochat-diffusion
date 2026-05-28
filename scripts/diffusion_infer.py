"""
Diffusion LLM Inference and Generation Script.

Usage:
    python -m scripts.diffusion_infer
    python -m scripts.diffusion_infer --prompt "Hello world"
"""

import os
import argparse
import json
import time

import torch
import torch.nn.functional as F

from nanochat_diffusion.diffusion_model import DiffusionModel
from nanochat_diffusion.diffusion_sampler import DiffusionSampler
from nanochat_diffusion.tokenizer import Tokenizer, UNK_TOKEN_ID
from nanochat_diffusion.common import print0, print_banner, get_base_dir, compute_init, autodetect_device_type
from nanochat_diffusion.checkpoint_manager import load_model

print_banner()


def setup_model(checkpoint_dir="", device="cuda", **kwargs):
    from nanochat_diffusion.checkpoint_manager import load_model
    model, tokenizer = load_model(
        model_name="diffusion",
        device=device,
        phase="train",
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )
    model.eval()
    return model, tokenizer


def diffusion_sampling(model, prompt, max_tokens=128, temperature=1.0, top_k=40, num_steps=20,
                       tokenizer=None):
    from nanochat_diffusion.diffusion_scheduler import generate_unmask_schedule

    device = model.get_device()
    config = model.config
    unk_id = config.unk_token_id
    num_diffusion_steps = config.num_diffusion_steps

    if isinstance(prompt, str):
        if tokenizer is not None:
            prompt_tokens = tokenizer.encode(prompt, prepend=False)
        else:
            prompt_tokens = [b for b in prompt.encode('utf-8')]
    else:
        prompt_tokens = list(prompt)

    seq_len = max_tokens + len(prompt_tokens)
    current_tokens = torch.full((1, seq_len), unk_id, dtype=torch.long, device=device)
    current_tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, device=device)

    is_determined = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
    is_determined[0, :len(prompt_tokens)] = True

    t_schedule = torch.linspace(
        num_diffusion_steps - 1, 0, num_steps, dtype=torch.long, device=device
    )

    unmask_schedule = generate_unmask_schedule(num_steps=num_steps)

    forbidden_ids = {0, unk_id}

    print0(f"Diffusion sampling: seq_len={seq_len}, prompt_len={len(prompt_tokens)}, steps={num_steps}")

    for step in range(num_steps):
        t = t_schedule[step].unsqueeze(0)

        with torch.no_grad():
            logits = model(current_tokens, t=t)

        undetermined = (~is_determined[0]).bool()
        unk_positions = torch.where(undetermined)[0]

        if len(unk_positions) == 0:
            print0("All tokens determined!")
            break

        unmask_frac = unmask_schedule[step]
        num_to_unmask = max(1, int(len(unk_positions) * unmask_frac))

        perm = torch.randperm(len(unk_positions), device=device)
        chosen_positions = unk_positions[perm[:num_to_unmask]]

        for pos in chosen_positions:
            logit = logits[0, pos].clone()
            for fid in forbidden_ids:
                if fid < len(logit):
                    logit[fid] = -1e10
            if temperature > 0:
                logit = logit / temperature
                if top_k > 0:
                    vals, _ = torch.topk(logit, min(top_k, logit.size(-1)))
                    logit[logit < vals[-1]] = -1e10
                probs = F.softmax(logit, dim=-1)
                if probs.sum() > 0:
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = torch.argmax(logits[0, pos]).item()
            else:
                next_token = torch.argmax(logit).item()
            current_tokens[0, pos] = next_token
            is_determined[0, pos] = True

        filled = is_determined.sum().item()
        print0(f"  Step {step+1}/{num_steps} (t={t.item():4d}): {filled}/{seq_len} tokens determined")

    print()
    return current_tokens[0]


def main():
    parser = argparse.ArgumentParser(description="Diffusion LLM Inference")

    parser.add_argument("--checkpoint-dir", type=str, default="", help="checkpoint directory")
    parser.add_argument("--checkpoint-step", type=str, default="latest", help="checkpoint step")
    parser.add_argument("--device", type=str, default="", help="device (auto=autodetect)")

    parser.add_argument("--seq-len", type=int, default=256, help="sequence length")
    parser.add_argument("--max-tokens", type=int, default=128, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="top-k filtering")
    parser.add_argument("--num-steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--prompt", type=str, default="", help="text prompt to continue")
    parser.add_argument("--output-file", type=str, default="", help="save output to file")

    args = parser.parse_args()

    device_type = args.device or autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    model, tokenizer = setup_model(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        device=device
    )

    print0(f"Device: {device}")
    print0(f"Checkpoint: {args.checkpoint_step}")

    if args.prompt:
        print0(f"Prompt: {args.prompt}")

        print0("=" * 80)
        print0("Diffusion Sampling")
        print0("=" * 80)
        generated = diffusion_sampling(
            model, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            num_steps=args.num_steps,
            tokenizer=tokenizer
        )

        decoded = tokenizer.decode(generated.tolist())
        print(f"\nGenerated text:\n{decoded}")

        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(f"Prompt: {args.prompt}\n\nGenerated:\n")
                f.write(decoded)
            print0(f"Output saved to {args.output_file}")

    else:
        print0("Running quick evaluation")
        for i in range(3):
            print0(f"\nSample {i+1}:")
            generated = diffusion_sampling(
                model, "Hello",
                max_tokens=64,
                temperature=0.7,
                top_k=30,
                num_steps=15
            )
            decoded = tokenizer.decode(generated.tolist())
            print(f"Generated: {decoded}")

    print0("=" * 80)
    print0("Inference complete!")
    print0("=" * 80)


if __name__ == "__main__":
    main()
