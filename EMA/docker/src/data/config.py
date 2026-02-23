"""
Data Configuration for TWAE-MMD 
======================================================

 configuration for antimicrobial peptide sequence processing.
This configuration handles the essential data processing parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


@dataclass
class DataConfig:
    """
    Simple data processing configuration for TWAE-MMD.
    
    This configuration manages essential aspects of data processing including
    file paths, preprocessing parameters, and tokenization settings.
    """
    
    # ===== Dataset Paths =====
    data_dir: str = "/workspace/TWAE_AMP_Generation/data"
    train_file: str = "/workspace/TWAE_AMP_Generation/data/raw/final_3_36_train.csv"
    test_file: str = "/workspace/TWAE_AMP_Generation/data/raw/final_3_36_test.csv"
    
    # Column names
    sequence_column: str = "sequence"
    label_column: str = "label"
    
    # ===== Sequence Parameters =====
    min_length: int = 3
    max_length: int = 36
    max_length_with_tokens: int = 37  # Including special tokens
    
    # Amino acids (20 standard)
    amino_acids: List[str] = field(default_factory=lambda: [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ])
    
    # ===== Tokenization Parameters =====
    vocab_size: int = 25  # 20 amino acids + 5 special tokens
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    
    # ===== Processing Parameters =====
    batch_size: int = 64
    validation_batch_size: int = 128
    test_batch_size: int = 128
    validation_split: float = 0.2
    
    # ===== Quality Control =====
    remove_duplicates: bool = True
    case_sensitive: bool = False
    strict_validation: bool = True
    
    # ===== Performance =====
    num_workers: int = 4
    prefetch_buffer_size: int = 2
    use_cache: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths are Path objects
        self.data_dir = Path(self.data_dir)
        self.train_file = Path(self.train_file)
        self.test_file = Path(self.test_file)
        
        # Validate sequence parameters
        if self.min_length <= 0:
            raise ValueError("min_length must be positive")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be >= min_length")
        if self.max_length_with_tokens < self.max_length:
            raise ValueError("max_length_with_tokens must be >= max_length")
        
        # Validate vocabulary
        if len(self.amino_acids) != 20:
            raise ValueError("Must have exactly 20 standard amino acids")
        if self.vocab_size < len(self.amino_acids) + 5:
            raise ValueError("vocab_size must be >= amino_acids + special_tokens")
        
        # Validate batch sizes
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.validation_split <= 0 or self.validation_split >= 1:
            raise ValueError("validation_split must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def get_data_config(config_name: str = "default") -> DataConfig:
    """
    Get data configuration by name.
    
    Args:
        config_name: Configuration name ("default", "small", "large")
        
    Returns:
        DataConfig instance
    """
    if config_name == "default":
        return DataConfig()
    
    elif config_name == "small":
        return DataConfig(
            batch_size=32,
            validation_batch_size=64,
            test_batch_size=64,
            num_workers=2
        )
    
    elif config_name == "large":
        return DataConfig(
            batch_size=128,
            validation_batch_size=256,
            test_batch_size=256,
            num_workers=8,
            prefetch_buffer_size=4
        )
    
    else:
        raise ValueError(f"Unknown config name: {config_name}")


# Default configuration instance
DEFAULT_CONFIG = DataConfig()

