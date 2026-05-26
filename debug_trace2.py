#!/usr/bin/env python3
"""Debug: add print statements to find where the training script hangs."""
import sys, os, subprocess

# Patch the script to add prints
script_path = "/tmp/nanochat_diffusion/scripts/diffusion_train.py"
with open(script_path, "r") as f:
    content = f.read()

# Add print after "Peak flops" line
lines = content.split("\n")
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if "peak_flops = get_peak_flops" in line:
        # Add trace after this line
        indent = len(line) - len(line.lstrip())
        new_lines.append(" " * indent + "print0('TRACE: get_peak_flops done')")
    if "if args.eval_only:" in line:
        indent = len(line) - len(line.lstrip())
        new_lines.append(" " * indent + "print0('TRACE: past eval_only check')")
    if "model.train()" in line and "model.train" in line:
        indent = len(line) - len(line.lstrip())
        new_lines.append(" " * indent + "print0('TRACE: model.train() done')")

patched = "\n".join(new_lines)

# Run the patched script
proc = subprocess.run(
    [sys.executable, "-c", f"exec('''{content}''')"],
    cwd="/tmp/nanochat_diffusion",
    capture_output=True, text=True, timeout=10
)
print(proc.stdout)
print(proc.stderr)
