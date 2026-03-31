# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__ #noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# --- Eager imports: needed for inference ---
from fairseq_signals.distributed import utils as distributed_utils
from fairseq_signals.logging import meters, metrics

sys.modules["fairseq_signals.distributed_utils"] = distributed_utils
sys.modules["fairseq_signals.meters"] = meters
sys.modules["fairseq_signals.metrics"] = metrics

# initialize hydra
from fairseq_signals.dataclass.initialize import hydra_init
hydra_init()

# Core model/module infrastructure (needed for inference)
import fairseq_signals.models # noqa
import fairseq_signals.modules # noqa
import fairseq_signals.distributed # noqa
import fairseq_signals.tasks # noqa
from fairseq_signals.utils import pdb

# --- Lazy imports: training-only modules ---
# These are only loaded when accessed, avoiding unnecessary overhead.

def __getattr__(name):
    """Lazy-load training modules on first access."""
    if name == "criterions":
        import fairseq_signals.criterions # noqa
        return fairseq_signals.criterions
    if name == "optim":
        import fairseq_signals.optim # noqa
        import fairseq_signals.optim.lr_scheduler # noqa
        return fairseq_signals.optim
    raise AttributeError(f"module 'fairseq_signals' has no attribute {name!r}")
