"""Data registry for TwinCar: parse Stanford Cars into labels + splits."""

from .config import StanfordPaths, load_dataset_config, resolve_stanford_paths
from .dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CarDataset,
    build_transforms,
)
from .label_maps import LabelMaps, load_label_maps
from .labels import (
    KNOWN_MAKES,
    ParsedClassName,
    parse_class_name,
    parse_year,
    split_make_model,
)
from .registry import (
    CSV_COLUMNS,
    LabelRecord,
    WriteResult,
    assign_splits,
    build_label_records,
    build_labels_yaml,
    generate_registry,
    write_csv,
    write_labels_yaml,
)
from .stanford import TrainAnnotation, load_class_names, load_train_annotations

__all__ = [
    "CSV_COLUMNS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "KNOWN_MAKES",
    "CarDataset",
    "LabelMaps",
    "LabelRecord",
    "ParsedClassName",
    "StanfordPaths",
    "TrainAnnotation",
    "WriteResult",
    "assign_splits",
    "build_label_records",
    "build_labels_yaml",
    "build_transforms",
    "generate_registry",
    "load_class_names",
    "load_dataset_config",
    "load_label_maps",
    "load_train_annotations",
    "parse_class_name",
    "parse_year",
    "resolve_stanford_paths",
    "split_make_model",
    "write_csv",
    "write_labels_yaml",
]
