import os
import random
import numpy as np
import torch


def lock_determinism(seed: int) -> None:
    """Pin all RNG sources; call BEFORE any CUDA context exists (i.e. first
    line of main(), right after argument parsing)."""
    # Must be set before cuBLAS initializes its workspace.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)  # defensive global pin; spatial RNG uses its own Generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True avoids a hard crash on ops without a deterministic kernel
    # (nothing in this codebase currently needs that escape hatch).
    torch.use_deterministic_algorithms(True, warn_only=True)

