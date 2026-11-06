from dataclasses import dataclass
from typing import List, Literal
from util.hparams import HyperParams
from util.globals import *

@dataclass
class AttnRouteHyperParams(HyperParams):
    # Method configuration
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    
    mom2_update_weight: float
    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str

    alpha: float
    beta: float
    maxdist: float
    lr: float
    
    cov_mode: str = "full"        # {"full", "none", "damped"}
    cov_damping: float = 1e-2     # only used when cov_mode == "damped"

    # ===== Reproducibility =====
    seed: int = 42

    rank_mode: str = "sve"    # 可选："sve", "fixed"
    fixed_rank: int = 1       # 仅在 rank_mode="fixed" 时生效



