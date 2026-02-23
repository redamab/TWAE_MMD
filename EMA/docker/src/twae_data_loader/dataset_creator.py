"""
Dataset Creator - Direct Fix for Silent Iteration Failure
=========================================================

This version completely avoids tf.py_function and uses direct TensorFlow operations
to ensure dataset iteration works correctly.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class TensorFlowDatasetCreator:
    """
    TWAE-MMD Dataset Creator with Direct TensorFlow Operations
    
    This version avoids tf.py_function completely and uses direct tensor operations
    to ensure reliable dataset iteration.
    """
    
    def __init__(self, tokenizer, config):
        """Initialize the dataset creator."""
        self.tokenizer = tokenizer
        self.config = config
        
    def create_dataset(self, sequences: List[str], labels: List[int], 
                      batch_size: int = 64, shuffle: bool = True,
                      cache: bool = True, prefetch: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with direct operations (no tf.py_function).
        
        Args:
            sequences: List of peptide sequences
            labels: List of corresponding labels
            batch_size: Batch size for training
            shuffle: Whether to shuffle the dataset
            cache: Whether to cache the dataset
            prefetch: Whether to prefetch batches
            
        Returns:
            tf.data.Dataset ready for training
        """
        logger.info(f"Creating TensorFlow dataset with {len(sequences)} sequences")
        
        # Pre-tokenize all sequences using the tokenizer directly
        logger.info("Pre-tokenizing all sequences...")
        tokenized_data = self._pre_tokenize_sequences(sequences)
        
        # Create dataset from pre-tokenized data
        logger.info("Creating dataset from pre-tokenized data...")
        dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask'],
            'labels': np.array(labels, dtype=np.int32)
        })
        
        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(10000, len(sequences)))
            
        # Batch the dataset
        logger.info("Creating batched dataset...")
        dataset = dataset.batch(batch_size)
        
        # Apply padding to batches
        dataset = dataset.map(
            self._pad_batch_direct,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply optimizations
        if cache:
            dataset = dataset.cache()
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
        logger.info(f"Dataset created successfully with batch size {batch_size}")
        return dataset
    
    def _pre_tokenize_sequences(self, sequences: List[str]) -> Dict[str, np.ndarray]:
        """
        Pre-tokenize all sequences using the tokenizer directly.
        
        This avoids tf.py_function completely by doing all tokenization upfront.
        """
        input_ids_list = []
        attention_mask_list = []
        
        for sequence in sequences:
            # Use the tokenizer directly (no TensorFlow operations)
            tokenized = self.tokenizer.encode(sequence)
            
            # Convert lists to numpy arrays
            input_ids = np.array(tokenized['input_ids'], dtype=np.int32)
            attention_mask = np.array(tokenized['attention_mask'], dtype=np.int32)
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        # Stack all sequences into arrays
        input_ids_array = np.stack(input_ids_list, axis=0)
        attention_mask_array = np.stack(attention_mask_list, axis=0)
        
        return {
            'input_ids': input_ids_array,
            'attention_mask': attention_mask_array
        }
    
    def _pad_batch_direct(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Pad batch using direct TensorFlow operations.
        
        This ensures all sequences in a batch have the same length.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Get current sequence length
        current_length = tf.shape(input_ids)[1]
        
        # Pad or truncate to max_length
        if current_length < self.config.max_length:
            # Pad sequences
            pad_length = self.config.max_length - current_length
            
            input_ids = tf.pad(
                input_ids,
                [[0, 0], [0, pad_length]],
                constant_values=0
            )
            
            attention_mask = tf.pad(
                attention_mask,
                [[0, 0], [0, pad_length]],
                constant_values=0
            )
        else:
            # Truncate sequences
            input_ids = input_ids[:, :self.config.max_length]
            attention_mask = attention_mask[:, :self.config.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def create_inference_dataset(self, sequences: List[str], 
                               batch_size: int = 128) -> tf.data.Dataset:
        """
        Create dataset for inference (no labels).
        
        Args:
            sequences: List of peptide sequences
            batch_size: Batch size for inference
            
        Returns:
            tf.data.Dataset ready for inference
        """
        logger.info(f"Creating inference dataset with {len(sequences)} sequences")
        
        # Pre-tokenize all sequences
        tokenized_data = self._pre_tokenize_sequences(sequences)
        
        # Create dataset from pre-tokenized data (no labels)
        dataset = tf.data.Dataset.from_tensor_slices({
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask']
        })
        
        # Batch the dataset
        dataset = dataset.batch(batch_size)
        
        # Apply padding to batches
        dataset = dataset.map(
            self._pad_inference_batch_direct,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply optimizations
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Inference dataset created successfully")
        return dataset
    
    def _pad_inference_batch_direct(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Pad inference batch using direct TensorFlow operations."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Get current sequence length
        current_length = tf.shape(input_ids)[1]
        
        # Pad or truncate to max_length
        if current_length < self.config.max_length:
            # Pad sequences
            pad_length = self.config.max_length - current_length
            
            input_ids = tf.pad(
                input_ids,
                [[0, 0], [0, pad_length]],
                constant_values=0
            )
            
            attention_mask = tf.pad(
                attention_mask,
                [[0, 0], [0, pad_length]],
                constant_values=0
            )
        else:
            # Truncate sequences
            input_ids = input_ids[:, :self.config.max_length]
            attention_mask = attention_mask[:, :self.config.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def get_dataset_element_spec(self, include_labels: bool = True) -> Dict[str, tf.TensorSpec]:
        """Get the element specification for the dataset."""
        spec = {
            'input_ids': tf.TensorSpec(shape=(None, self.config.max_length), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None, self.config.max_length), dtype=tf.int32)
        }
        
        if include_labels:
            spec['labels'] = tf.TensorSpec(shape=(None,), dtype=tf.int32)
            
        return spec

def create_tensorflow_datasets(sequences_train: List[str], labels_train: List[int],
                              sequences_val: List[str], labels_val: List[int],
                              tokenizer, config, 
                              train_batch_size: int = 64, val_batch_size: int = 128) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create training and validation TensorFlow datasets.
    
    Args:
        sequences_train: Training sequences
        labels_train: Training labels
        sequences_val: Validation sequences  
        labels_val: Validation labels
        tokenizer: Tokenizer instance
        config: Configuration object
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create dataset creator
    creator = TensorFlowDatasetCreator(tokenizer, config)
    
    # Create training dataset
    train_dataset = creator.create_dataset(
        sequences=sequences_train,
        labels=labels_train,
        batch_size=train_batch_size,
        shuffle=True,
        cache=True,
        prefetch=True
    )
    
    # Create validation dataset
    val_dataset = creator.create_dataset(
        sequences=sequences_val,
        labels=labels_val,
        batch_size=val_batch_size,
        shuffle=False,
        cache=True,
        prefetch=True
    )
    
    return train_dataset, val_dataset




