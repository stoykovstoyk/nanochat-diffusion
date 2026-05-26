#!/usr/bin/env python3
"""Test: run training for 2 steps and verify it completes."""
import sys
import os
import signal
import subprocess

script = "/tmp/nanochat_diffusion/scripts/diffusion_train.py"
cmd = [sys.executable, "-m", "scripts.diffusion_train",
       "--num-cpus", "2",
       "--depth", "4",
       "--max-seq-len", "256",
       "--device-type", "cpu",
       "--device-batch-size", "4",
       "--num-iterations", "2",
       "--eval-iters", "1",
       "--save-every", "10"]

print(f"Running: {' '.join(cmd)}")
proc = subprocess.run(cmd, cwd="/tmp/nanochat_diffusion", capture_output=True, text=True, timeout=30)
print("STDOUT:")
print(proc.stdout)
print("\nSTDERR:")
print(proc.stderr)
print(f"\nExit code: {proc.returncode}")
