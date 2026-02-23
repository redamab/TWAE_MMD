"""
Data Loader Configuration for TWAE-MMD
======================================

Configuration management for the streamlined data loader.
Optimized for real AMP dataset training.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class DataLoaderConfig:
    """
    Configuration for TWAE-MMD data loader.
    
    Optimized for your real datasets:
    - 63K training sequences
    - 11K validation sequences
    - Sequence length: 3-36 amino acids
    """
    
    # ===== Dataset Paths =====
    train_csv_path: str = "/workspace/TWAE_AMP_Generation/data/raw/final_3_36_train.csv"
    val_csv_path: str = "/workspace/TWAE_AMP_Generation/data/raw/final_3_36_validation.csv"
    
    # ===== Batch Configuration =====
    train_batch_size: int = 64          # Training batch size
    val_batch_size: int = 128           # Validation batch size
    
    # ===== Sequence Configuration =====
    max_length: int = 37                # Maximum sequence length (36 + special tokens)
    min_length: int = 3                 # Minimum sequence length
    vocab_size: int = 25                # Vocabulary size (20 AA + 5 special)
    
    # ===== Data Processing =====
    shuffle_buffer_size: int = 10000    # Shuffle buffer size
    prefetch_buffer_size: int = 2       # Prefetch buffer size
    num_parallel_calls: int = 4         # Parallel processing calls
    
    # ===== Memory Optimization =====
    cache_dataset: bool = True          # Cache processed dataset
    drop_remainder: bool = True         # Drop incomplete batches
    
    # ===== CSV Processing =====
    sequence_column: str = "sequence"   # Name of sequence column
    label_column: str = "label"         # Name of label column
    skip_header: bool = True            # Skip CSV header
    
    # ===== Validation =====
    validate_sequences: bool = True     # Validate sequence format
    remove_invalid: bool = True         # Remove invalid sequences
    
    # ===== Performance =====
    use_multiprocessing: bool = True    # Use multiprocessing
    buffer_size: int = 1000            # I/O buffer size
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate batch sizes
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.val_batch_size <= 0:
            raise ValueError("val_batch_size must be positive")
            
        # Validate sequence parameters
        if self.max_length <= self.min_length:
            raise ValueError("max_length must be greater than min_length")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
            
        # Validate paths (if they exist)
        if self.train_csv_path and Path(self.train_csv_path).suffix != '.csv':
            raise ValueError("train_csv_path must be a CSV file")
        if self.val_csv_path and Path(self.val_csv_path).suffix != '.csv':
            raise ValueError("val_csv_path must be a CSV file")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataLoaderConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Predefined configurations
DEFAULT_CONFIG = DataLoaderConfig()

FAST_CONFIG = DataLoaderConfig(
    train_batch_size=128,
    val_batch_size=256,
    shuffle_buffer_size=5000,
    prefetch_buffer_size=4,
    num_parallel_calls=8,
    cache_dataset=True
)

MEMORY_EFFICIENT_CONFIG = DataLoaderConfig(
    train_batch_size=32,
    val_batch_size=64,
    shuffle_buffer_size=1000,
    prefetch_buffer_size=1,
    num_parallel_calls=2,
    cache_dataset=False
)

HIGH_THROUGHPUT_CONFIG = DataLoaderConfig(
    train_batch_size=256,
    val_batch_size=512,
    shuffle_buffer_size=20000,
    prefetch_buffer_size=8,
    num_parallel_calls=16,
    cache_dataset=True,
    drop_remainder=True
)

# Configuration registry
CONFIGS = {
    'default': DEFAULT_CONFIG,
    'fast': FAST_CONFIG,
    'memory_efficient': MEMORY_EFFICIENT_CONFIG,
    'high_throughput': HIGH_THROUGHPUT_CONFIG
}


def get_default_config() -> DataLoaderConfig:
    """Get default data loader configuration."""
    return DEFAULT_CONFIG


def get_config(name: str) -> DataLoaderConfig:
    """
    Get predefined configuration by name.
    
    Args:
        name: Configuration name ('default', 'fast', 'memory_efficient', 'high_throughput')
        
    Returns:
        DataLoaderConfig instance
    """
    if name not in CONFIGS:
        available = ', '.join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    return CONFIGS[name]


def create_custom_config(
    train_csv_path: str,
    val_csv_path: str,
    train_batch_size: int = 64,
    val_batch_size: int = 128,
    **kwargs
) -> DataLoaderConfig:
    """
    Create custom configuration with specified paths and batch sizes.
    
    Args:
        train_csv_path: Path to training CSV file
        val_csv_path: Path to validation CSV file
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        **kwargs: Additional configuration parameters
        
    Returns:
        DataLoaderConfig instance
    """
    return DataLoaderConfig(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        **kwargs
    )

