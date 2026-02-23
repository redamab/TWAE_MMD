"""
TWAE-MMD Streamlined Data Loader
================================

Production-ready data loader for TWAE-MMD model training with real AMP datasets.

This module provides:
- Direct CSV dataset loading
- TensorFlow dataset creation
- TWAE-MMD model integration
- Optimized batching and preprocessing
- Memory-efficient data pipeline


Author: Reda Mabrouki

"""

from .data_loader import (
    TWAEMMDDataLoader,
    create_data_loader,
    load_amp_datasets
)

from .dataset_creator import (
    TensorFlowDatasetCreator,
    create_tensorflow_datasets
)

from .config import (
    DataLoaderConfig,
    get_default_config,
    get_config,
    create_custom_config
)

from .utils import (
    validate_csv_format,
    analyze_dataset,
    print_dataset_info
)

# Main factory function for easy usage
def create_complete_data_pipeline(
    train_csv_path: str,
    val_csv_path: str,
    tokenizer=None,
    preprocessor=None,
    config=None
):
    """
    Create complete data pipeline for TWAE-MMD training.
    
    Args:
        train_csv_path: Path to training CSV file
        val_csv_path: Path to validation CSV file
        tokenizer: Peptide tokenizer (optional, will create if None)
        preprocessor: Peptide preprocessor (optional, will create if None)
        config: Data loader configuration (optional, will use default if None)
        
    Returns:
        Tuple of (train_dataset, val_dataset, data_loader)
    """
    # Create data loader
    data_loader = create_data_loader(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        config=config
    )
    
    # Load datasets
    train_dataset, val_dataset = data_loader.create_training_datasets(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path
    )
    
    return train_dataset, val_dataset, data_loader

# Export main components
__all__ = [
    # Main classes
    'TWAEMMDDataLoader',
    'TensorFlowDatasetCreator', 
    'DataLoaderConfig',
    
    # Factory functions
    'create_data_loader',
    'create_tensorflow_datasets',
    'create_complete_data_pipeline',
    'load_amp_datasets',
    'get_default_config',
    'get_config',
    'create_custom_config',
    
    # Utilities
    'validate_csv_format',
    'analyze_dataset',
    'print_dataset_info'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "TWAE-MMD Data Pipeline Team"
__description__ = "Streamlined data loader for TWAE-MMD model with real AMP datasets"

# Quick usage example
USAGE_EXAMPLE = '''
# Quick Usage Example:
from twae_data_loader import create_complete_data_pipeline

# Create complete pipeline
train_dataset, val_dataset, data_loader = create_complete_data_pipeline(
    train_csv_path="/workspace/TWAE_AMP_Generation/data/raw/final_3_36_train.csv",
    val_csv_path="/workspace/TWAE_AMP_Generation/data/raw/final_3_36_validation.csv"
)

# Ready for TWAE-MMD training!
for batch in train_dataset:
    # batch contains: input_ids, attention_mask, labels
    outputs = model(batch['input_ids'], training=True)
    # ... training step ...
'''

