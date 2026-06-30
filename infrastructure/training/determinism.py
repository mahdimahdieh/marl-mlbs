import os
import random
import numpy as np
import torch


def lock_determinism(seed: int) -> None:
    """
    Call this BEFORE any CUDA context exists — i.e. before constructing any
    nn.Module or moving a tensor to device. Safest call site: first line of
    main()/run_inference(), immediately after argument parsing.
    """
    # Must be set before cuBLAS initializes its workspace; if a CUDA call has
    # already happened in this process, this line is too late — see note below.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)        # defensive global pin only — spatial RNG below
                                 # still uses its own local Generator, unaffected
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True avoids a hard crash on any op without a deterministic
    # kernel; nothing in this codebase currently needs that escape hatch
    # (pure Linear/Tanh/Categorical), so you can tighten to warn_only=False
    # once you've confirmed no PyWiSim eval-mode op trips it.
    torch.use_deterministic_algorithms(True, warn_only=True)