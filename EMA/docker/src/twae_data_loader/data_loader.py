"""
Main Data Loader for TWAE-MMD
=============================

Streamlined data loader for real AMP datasets with TWAE-MMD model integration.
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path

from .config import DataLoaderConfig, get_default_config
from .dataset_creator import TensorFlowDatasetCreator
from .utils import validate_csv_format, analyze_dataset, print_dataset_info


class TWAEMMDDataLoader:
    """
    Streamlined data loader for TWAE-MMD model training.
    
    This loader handles:
    - Loading real AMP CSV datasets
    - Integration with existing tokenizer/preprocessor
    - TensorFlow dataset creation
    - Optimized batching and preprocessing
    - Memory-efficient data pipeline
    """
    
    def __init__(self,
                 tokenizer=None,
                 preprocessor=None,
                 config: Optional[DataLoaderConfig] = None):
        """
        Initialize TWAE-MMD data loader.
        
        Args:
            tokenizer: Peptide tokenizer (will create if None)
            preprocessor: Peptide preprocessor (will create if None)
            config: Data loader configuration (will use default if None)
        """
        self.config = config or get_default_config()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer and preprocessor
        self._setup_components(tokenizer, preprocessor)
        
        # Create TensorFlow dataset creator
        self.dataset_creator = TensorFlowDatasetCreator(
            tokenizer=self.tokenizer,
            config=self.config
        )
        
        # Statistics
        self.stats = {
            'train_sequences': 0,
            'val_sequences': 0,
            'total_sequences': 0,
            'train_batches': 0,
            'val_batches': 0
        }
    
    def _setup_components(self, tokenizer, preprocessor):
        """Setup tokenizer and preprocessor components."""
        if tokenizer is None:
            # Import and create tokenizer
            try:
                from data import create_tokenizer
                self.tokenizer = create_tokenizer(
                    vocab_size=self.config.vocab_size,
                    max_length=self.config.max_length
                )
                self.logger.info("Created default tokenizer")
            except ImportError:
                raise ImportError("Cannot import tokenizer. Please provide tokenizer or ensure 'data' module is available.")
        else:
            self.tokenizer = tokenizer
            self.logger.info("Using provided tokenizer")
        
        if preprocessor is None:
            # Import and create preprocessor
            try:
                from data import create_preprocessor
                self.preprocessor = create_preprocessor(
                    min_length=self.config.min_length,
                    max_length=self.config.max_length - 1  # Subtract 1 for special tokens
                )
                self.logger.info("Created default preprocessor")
            except ImportError:
                raise ImportError("Cannot import preprocessor. Please provide preprocessor or ensure 'data' module is available.")
        else:
            self.preprocessor = preprocessor
            self.logger.info("Using provided preprocessor")
    
    def load_csv_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load AMP dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with sequences and labels
        """
        self.logger.info(f"Loading dataset from: {csv_path}")
        
        # Validate file exists
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Validate CSV format
        validate_csv_format(csv_path, self.config)
        
        # Load CSV
        df = pd.read_csv(
            csv_path,
            usecols=[self.config.sequence_column, self.config.label_column]
        )
        
        self.logger.info(f"Loaded {len(df)} sequences from {csv_path}")
        
        # Validate and preprocess if enabled
        if self.config.validate_sequences:
            df = self._validate_and_clean_dataset(df)
        
        return df
    
    def _validate_and_clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean dataset using preprocessor.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Validating and cleaning dataset...")
        
        initial_count = len(df)
        
        # Use preprocessor to clean sequences
        cleaned_df = self.preprocessor.preprocess_dataframe(
            df,
            sequence_column=self.config.sequence_column,
            label_column=self.config.label_column
        )
        
        final_count = len(cleaned_df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} invalid sequences ({removed_count/initial_count*100:.1f}%)")
        
        return cleaned_df
    
    def create_tensorflow_dataset(self,
                                csv_path: str,
                                batch_size: Optional[int] = None,
                                shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from CSV file.
        
        Args:
            csv_path: Path to CSV file
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle dataset
            
        Returns:
            TensorFlow dataset ready for training
        """
        # Load CSV data
        df = self.load_csv_dataset(csv_path)
        
        # Extract sequences and labels
        sequences = df[self.config.sequence_column].tolist()
        labels = df[self.config.label_column].tolist()
        
        # Create TensorFlow dataset
        dataset = self.dataset_creator.create_dataset(
            sequences=sequences,
            labels=labels,
            batch_size=batch_size,
            shuffle=shuffle
        )
        
        # Update statistics
        num_sequences = len(sequences)
        effective_batch_size = batch_size or (
            self.config.train_batch_size if shuffle else self.config.val_batch_size
        )
        num_batches = (num_sequences + effective_batch_size - 1) // effective_batch_size
        
        if shuffle:
            self.stats['train_sequences'] = num_sequences
            self.stats['train_batches'] = num_batches
        else:
            self.stats['val_sequences'] = num_sequences
            self.stats['val_batches'] = num_batches
        
        self.stats['total_sequences'] = self.stats['train_sequences'] + self.stats['val_sequences']
        
        return dataset
    
    def create_training_datasets(self,
                               train_csv_path: Optional[str] = None,
                               val_csv_path: Optional[str] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Create training and validation datasets.
        
        Args:
            train_csv_path: Path to training CSV (uses config default if None)
            val_csv_path: Path to validation CSV (uses config default if None)
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Use config paths if not provided
        train_path = train_csv_path or self.config.train_csv_path
        val_path = val_csv_path or self.config.val_csv_path
        
        self.logger.info("Creating training datasets...")
        self.logger.info(f"Training data: {train_path}")
        self.logger.info(f"Validation data: {val_path}")
        
        # Create training dataset
        train_dataset = self.create_tensorflow_dataset(
            csv_path=train_path,
            batch_size=self.config.train_batch_size,
            shuffle=True
        )
        
        # Create validation dataset
        val_dataset = self.create_tensorflow_dataset(
            csv_path=val_path,
            batch_size=self.config.val_batch_size,
            shuffle=False
        )
        
        # Print dataset information
        self.print_dataset_summary()
        
        return train_dataset, val_dataset
    
    def analyze_datasets(self,
                        train_csv_path: Optional[str] = None,
                        val_csv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze training and validation datasets.
        
        Args:
            train_csv_path: Path to training CSV (uses config default if None)
            val_csv_path: Path to validation CSV (uses config default if None)
            
        Returns:
            Analysis results
        """
        train_path = train_csv_path or self.config.train_csv_path
        val_path = val_csv_path or self.config.val_csv_path
        
        # Load datasets
        train_df = self.load_csv_dataset(train_path)
        val_df = self.load_csv_dataset(val_path)
        
        # Analyze datasets
        train_analysis = analyze_dataset(train_df, self.config.sequence_column, self.config.label_column)
        val_analysis = analyze_dataset(val_df, self.config.sequence_column, self.config.label_column)
        
        # Combined analysis
        combined_analysis = {
            'train': train_analysis,
            'validation': val_analysis,
            'total_sequences': train_analysis['total_sequences'] + val_analysis['total_sequences'],
            'train_ratio': train_analysis['total_sequences'] / (train_analysis['total_sequences'] + val_analysis['total_sequences']),
            'val_ratio': val_analysis['total_sequences'] / (train_analysis['total_sequences'] + val_analysis['total_sequences'])
        }
        
        return combined_analysis
    
    def print_dataset_summary(self):
        """Print dataset summary information."""
        print("\n" + "="*60)
        print("TWAE-MMD DATASET SUMMARY")
        print("="*60)
        print(f"Training sequences:     {self.stats['train_sequences']:,}")
        print(f"Validation sequences:   {self.stats['val_sequences']:,}")
        print(f"Total sequences:        {self.stats['total_sequences']:,}")
        print(f"Training batches:       {self.stats['train_batches']:,}")
        print(f"Validation batches:     {self.stats['val_batches']:,}")
        print(f"Train batch size:       {self.config.train_batch_size}")
        print(f"Validation batch size:  {self.config.val_batch_size}")
        print(f"Max sequence length:    {self.config.max_length}")
        print(f"Vocabulary size:        {self.config.vocab_size}")
        print("="*60)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'config': self.config.to_dict(),
            'stats': self.stats.copy(),
            'tokenizer_vocab_size': getattr(self.tokenizer, 'vocab_size', None),
            'preprocessor_settings': {
                'min_length': getattr(self.preprocessor, 'min_length', None),
                'max_length': getattr(self.preprocessor, 'max_length', None)
            }
        }


def create_data_loader(tokenizer=None,
                      preprocessor=None,
                      config: Optional[DataLoaderConfig] = None) -> TWAEMMDDataLoader:
    """
    Create TWAE-MMD data loader with default settings.
    
    Args:
        tokenizer: Peptide tokenizer (optional)
        preprocessor: Peptide preprocessor (optional)
        config: Data loader configuration (optional)
        
    Returns:
        TWAEMMDDataLoader instance
    """
    return TWAEMMDDataLoader(
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        config=config
    )


def load_amp_datasets(train_csv_path: str,
                     val_csv_path: str,
                     tokenizer=None,
                     preprocessor=None,
                     config: Optional[DataLoaderConfig] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, TWAEMMDDataLoader]:
    """
    Load AMP datasets and create TensorFlow datasets.
    
    Args:
        train_csv_path: Path to training CSV file
        val_csv_path: Path to validation CSV file
        tokenizer: Peptide tokenizer (optional)
        preprocessor: Peptide preprocessor (optional)
        config: Data loader configuration (optional)
        
    Returns:
        Tuple of (train_dataset, val_dataset, data_loader)
    """
    # Create data loader
    data_loader = create_data_loader(tokenizer, preprocessor, config)
    
    # Create datasets
    train_dataset, val_dataset = data_loader.create_training_datasets(
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path
    )
    
    return train_dataset, val_dataset, data_loader

