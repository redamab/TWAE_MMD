"""
TWAE-MMD Data Module - Fixed and Simplified
==========================================

Simple, robust data processing pipeline for Transformer Wasserstein Autoencoder 
with Maximum Mean Discrepancy (TWAE-MMD) for antimicrobial peptide generation.

This module provides only essential components:
- Peptide tokenization and encoding
- Data preprocessing 
- Configuration management

Author: TWAE-MMD Data Pipeline
Date: 2025-09-13
Version: 2.0.0 - Fixed and Simplified
"""

# Core imports
from .tokenizer import (
    PeptideTokenizer,
    create_tokenizer
)

from .preprocessor import (
    PeptidePreprocessor,
    create_preprocessor
)

from .config import (
    DataConfig,
    get_data_config
)

# Create aliases for easier use
AMPTokenizer = PeptideTokenizer
AMPPreprocessor = PeptidePreprocessor  

# Simple pipeline creation function
def create_simple_pipeline(vocab_size: int = 25,
                          max_length: int = 37,
                          min_length: int = 3,
                          max_seq_length: int = 36):
    """
    Create simple data pipeline with essential components.
    
    Args:
        vocab_size: Vocabulary size for tokenizer
        max_length: Maximum sequence length (including special tokens)
        min_length: Minimum sequence length
        max_seq_length: Maximum sequence length (before adding special tokens)
        
    Returns:
        Tuple of (tokenizer, preprocessor)
    """
    # Create tokenizer
    tokenizer = create_tokenizer(
        vocab_size=vocab_size,
        max_length=max_length
    )
    
    # Create preprocessor
    preprocessor = create_preprocessor(
        min_length=min_length,
        max_length=max_seq_length
    )
    
    return tokenizer, preprocessor

# Export main classes and functions
__all__ = [
    # Core classes
    'PeptideTokenizer',
    'PeptidePreprocessor',
    'DataConfig',
    
    # Aliases for easier use
    'AMPTokenizer',
    'AMPPreprocessor',
    
    # Factory functions
    'create_tokenizer',
    'create_preprocessor',
    'create_simple_pipeline',
    'get_data_config',
]

# Module metadata
__version__ = "2.0.0"
__author__ = "TWAE-MMD Data Pipeline - Fixed"
__description__ = "Simple, robust data processing pipeline for TWAE-MMD antimicrobial peptide generation"

