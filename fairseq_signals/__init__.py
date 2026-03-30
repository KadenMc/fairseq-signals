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
sys.modules["fairseq_signals.distributed_utils"] = distributed_utils

# --- Eager imports: core model/module infrastructure ---
import fairseq_signals.models # noqa
import fairseq_signals.modules # noqa
import fairseq_signals.distributed # noqa
from fairseq_signals.utils import pdb

# --- Lazy imports: only loaded when accessed ---
# These pull in heavy/unnecessary deps (sklearn, hydra, etc.)
# and are only needed for training, evaluation, or Hydra CLI usage.

def _lazy_init():
    """Initialize Hydra and register training components.
    Called automatically when training modules are first accessed,
    or can be called explicitly if needed."""
    from fairseq_signals.logging import meters, metrics
    sys.modules["fairseq_signals.meters"] = meters
    sys.modules["fairseq_signals.metrics"] = metrics

    from fairseq_signals.dataclass.initialize import hydra_init
    hydra_init()

    import fairseq_signals.criterions # noqa
    import fairseq_signals.optim # noqa
    import fairseq_signals.optim.lr_scheduler # noqa
    import fairseq_signals.tasks # noqa

    _lazy_init._done = True

_lazy_init._done = False


def __getattr__(name):
    """Lazy-load training modules on first access."""
    lazy_modules = {"criterions", "optim", "tasks", "meters", "metrics"}
    if name in lazy_modules:
        if not _lazy_init._done:
            _lazy_init()
        return sys.modules.get(f"fairseq_signals.{name}")
    raise AttributeError(f"module 'fairseq_signals' has no attribute {name!r}")
