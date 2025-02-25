from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.ooc_datasets import (
    OOCDataset,
)
from lavis.common.registry import registry
 

import os
import logging
import warnings

import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized, is_main_process

@registry.register_builder("ooc")
class OOCBuilder(BaseDatasetBuilder):
    train_dataset_cls = OOCDataset
    eval_dataset_cls = OOCDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ooc/defaults_caption.yaml",
    }
    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets