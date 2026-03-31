# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys

from .dataset import BaseDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    ShardedIterator
)

# Dataset classes are lazy-imported to avoid pulling in heavy dependencies
# (wfdb, transformers, etc.) when only data_utils is needed (e.g., for inference).
_LAZY_IMPORTS = {
    "FileECGDataset": ".ecg.raw_ecg_dataset",
    "PathECGDataset": ".ecg.raw_ecg_dataset",
    "CMSCECGDataset": ".ecg.cmsc_ecg_dataset",
    "PerturbECGDataset": ".ecg.perturb_ecg_dataset",
    "ThreeKGECGDataset": ".ecg.perturb_ecg_dataset",
    "IdentificationECGDataset": ".ecg.identification_ecg_dataset",
    "SegmentationECGDataset": ".ecg.segmentation_ecg_dataset",
    "FileECGQADataset": ".ecg_text.ecg_qa_dataset",
    "FileECGTextDataset": ".ecg_text.ecg_text_dataset",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        obj = getattr(module, name)
        globals()[name] = obj  # cache for subsequent access
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseDataset",
    "CountingIterator",
    "EpochBatchIterator",
    "ShardedIterator",
    "FileECGDataset",
    "PathECGDataset",
    "CMSCECGDataset",
    "PerturbECGDataset",
    "ThreeKGECGDataset",
    "IdentificationECGDataset",
    "SegmentationECGDataset",
    "FileECGQADataset",
    "FileECGTextDataset",
]
