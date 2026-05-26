"""
Common utilities for nanochat_diffusion.
Adapted from karpathy/nanochat.
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat_diffusion")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"
    if os.path.exists(file_path):
        return file_path
    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")
        if postprocess_fn is not None:
            postprocess_fn(file_path)
    return file_path

def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def autodetect_device_type():
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"):
    assert device_type in ["cuda", "mps", "cpu"]
    if device_type == "cuda":
        assert torch.cuda.is_available()
    if device_type == "mps":
        assert torch.backends.mps.is_available()
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")
    is_ddp_requested_flag, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested_flag and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)
    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")
    return is_ddp_requested_flag, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    def __init__(self): pass
    def log(self, *args, **kwargs): pass
    def finish(self): pass

def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()
    _TABLE = (
        (["h200", "nvl"], 836e12), (["h200", "pcie"], 836e12), (["h200"], 989e12),
        (["h100", "nvl"], 835e12), (["h100", "pcie"], 756e12), (["h100"], 989e12),
        (["a100"], 312e12), (["a800"], 312e12), (["a40"], 149.7e12), (["a30"], 165e12),
        (["l40s"], 362e12), (["l4"], 121e12), (["3090"], 71e12), (["4090"], 165.2e12),
    )
    for patterns, flops in _TABLE:
        if all(p in name for p in patterns):
            return flops
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
