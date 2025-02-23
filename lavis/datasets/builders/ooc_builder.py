from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.ooc_datasets import (
    OOCDataset,
)
@registry.register_builder("ooc")
class OOCBuilder(BaseDatasetBuilder):
    train_dataset_cls = OOCDataset
    eval_dataset_cls = OOCDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/ooc/defaults_caption.yaml",
    }