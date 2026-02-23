"""
Utility Functions for TWAE-MMD Data Loader
==========================================

Helper functions for dataset validation, analysis, and debugging.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from collections import Counter

from .config import DataLoaderConfig


def validate_csv_format(csv_path: str, config: DataLoaderConfig) -> bool:
    """
    Validate CSV file format and structure.
    
    Args:
        csv_path: Path to CSV file
        config: Data loader configuration
        
    Returns:
        True if valid, raises exception if invalid
    """
    logger = logging.getLogger(__name__)
    
    # Check file exists
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Check file extension
    if not csv_path.endswith('.csv'):
        raise ValueError(f"File must be CSV format: {csv_path}")
    
    try:
        # Read first few rows to validate structure
        df_sample = pd.read_csv(csv_path, nrows=5)
        
        # Check required columns exist
        required_columns = [config.sequence_column, config.label_column]
        missing_columns = [col for col in required_columns if col not in df_sample.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns {missing_columns} in {csv_path}. "
                           f"Available columns: {list(df_sample.columns)}")
        
        # Check data types
        if not df_sample[config.sequence_column].dtype == 'object':
            logger.warning(f"Sequence column should be string type in {csv_path}")
        
        if not pd.api.types.is_numeric_dtype(df_sample[config.label_column]):
            logger.warning(f"Label column should be numeric type in {csv_path}")
        
        logger.info(f"CSV format validation passed for {csv_path}")
        return True
        
    except Exception as e:
        raise ValueError(f"Error validating CSV format for {csv_path}: {str(e)}")


def analyze_dataset(df: pd.DataFrame, 
                   sequence_column: str = 'sequence',
                   label_column: str = 'label') -> Dict[str, Any]:
    """
    Analyze dataset characteristics.
    
    Args:
        df: DataFrame with sequences and labels
        sequence_column: Name of sequence column
        label_column: Name of label column
        
    Returns:
        Dictionary with analysis results
    """
    sequences = df[sequence_column].tolist()
    labels = df[label_column].tolist()
    
    # Basic statistics
    total_sequences = len(sequences)
    
    # Length analysis
    lengths = [len(seq) for seq in sequences]
    length_stats = {
        'min': min(lengths),
        'max': max(lengths),
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths)
    }
    
    # Length distribution
    length_counts = Counter(lengths)
    
    # Label analysis
    label_counts = Counter(labels)
    label_distribution = {
        label: count / total_sequences 
        for label, count in label_counts.items()
    }
    
    # Amino acid analysis
    all_sequences = ''.join(sequences)
    aa_counts = Counter(all_sequences)
    total_amino_acids = len(all_sequences)
    
    aa_frequencies = {
        aa: count / total_amino_acids 
        for aa, count in aa_counts.items()
    }
    
    # Sequence quality checks
    valid_amino_acids = set('ARNDCQEGHILKMFPSTWYV')
    invalid_chars = set(all_sequences) - valid_amino_acids
    
    # Duplicate analysis
    unique_sequences = len(set(sequences))
    duplicate_count = total_sequences - unique_sequences
    
    return {
        'total_sequences': total_sequences,
        'unique_sequences': unique_sequences,
        'duplicate_count': duplicate_count,
        'duplicate_rate': duplicate_count / total_sequences if total_sequences > 0 else 0,
        
        'length_stats': length_stats,
        'length_distribution': dict(length_counts),
        
        'label_counts': dict(label_counts),
        'label_distribution': label_distribution,
        'class_balance': min(label_distribution.values()) / max(label_distribution.values()) if label_distribution else 0,
        
        'amino_acid_counts': dict(aa_counts),
        'amino_acid_frequencies': aa_frequencies,
        'total_amino_acids': total_amino_acids,
        'unique_amino_acids': len(aa_counts),
        
        'invalid_characters': list(invalid_chars),
        'has_invalid_chars': len(invalid_chars) > 0,
        
        'quality_score': _calculate_quality_score(
            total_sequences, duplicate_count, len(invalid_chars), label_distribution
        )
    }


def _calculate_quality_score(total_sequences: int,
                           duplicate_count: int,
                           invalid_char_count: int,
                           label_distribution: Dict[int, float]) -> float:
    """
    Calculate dataset quality score (0-1, higher is better).
    
    Args:
        total_sequences: Total number of sequences
        duplicate_count: Number of duplicate sequences
        invalid_char_count: Number of invalid characters
        label_distribution: Distribution of labels
        
    Returns:
        Quality score between 0 and 1
    """
    if total_sequences == 0:
        return 0.0
    
    # Penalize duplicates
    duplicate_penalty = duplicate_count / total_sequences
    
    # Penalize invalid characters
    invalid_penalty = min(invalid_char_count / total_sequences, 1.0)
    
    # Reward balanced classes
    if len(label_distribution) > 1:
        class_balance = min(label_distribution.values()) / max(label_distribution.values())
    else:
        class_balance = 1.0
    
    # Calculate quality score
    quality_score = (1.0 - duplicate_penalty) * (1.0 - invalid_penalty) * class_balance
    
    return max(0.0, min(1.0, quality_score))


def print_dataset_info(analysis: Dict[str, Any], dataset_name: str = "Dataset"):
    """
    Print formatted dataset analysis information.
    
    Args:
        analysis: Analysis results from analyze_dataset
        dataset_name: Name of the dataset for display
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} ANALYSIS")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Total sequences:        {analysis['total_sequences']:,}")
    print(f"Unique sequences:       {analysis['unique_sequences']:,}")
    print(f"Duplicate sequences:    {analysis['duplicate_count']:,} ({analysis['duplicate_rate']*100:.1f}%)")
    print(f"Quality score:          {analysis['quality_score']:.3f}")
    
    # Length statistics
    length_stats = analysis['length_stats']
    print(f"\nSEQUENCE LENGTH STATISTICS:")
    print(f"Min length:             {length_stats['min']}")
    print(f"Max length:             {length_stats['max']}")
    print(f"Mean length:            {length_stats['mean']:.1f}")
    print(f"Median length:          {length_stats['median']:.1f}")
    print(f"Std deviation:          {length_stats['std']:.1f}")
    
    # Label distribution
    print(f"\nLABEL DISTRIBUTION:")
    for label, count in analysis['label_counts'].items():
        percentage = analysis['label_distribution'][label] * 100
        label_name = "AMP" if label == 1 else "non-AMP"
        print(f"{label_name} (label {label}):      {count:,} ({percentage:.1f}%)")
    print(f"Class balance:          {analysis['class_balance']:.3f}")
    
    # Amino acid info
    print(f"\nAMINO ACID STATISTICS:")
    print(f"Total amino acids:      {analysis['total_amino_acids']:,}")
    print(f"Unique amino acids:     {analysis['unique_amino_acids']}")
    
    # Top amino acids
    top_aas = sorted(analysis['amino_acid_frequencies'].items(), 
                    key=lambda x: x[1], reverse=True)[:5]
    print(f"Most frequent AAs:      {', '.join([f'{aa}({freq*100:.1f}%)' for aa, freq in top_aas])}")
    
    # Quality issues
    if analysis['has_invalid_chars']:
        print(f"\nQUALITY ISSUES:")
        print(f"Invalid characters:     {', '.join(analysis['invalid_characters'])}")
    
    print(f"{'='*60}")


def compare_datasets(train_analysis: Dict[str, Any], 
                    val_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare training and validation datasets.
    
    Args:
        train_analysis: Training dataset analysis
        val_analysis: Validation dataset analysis
        
    Returns:
        Comparison results
    """
    total_sequences = train_analysis['total_sequences'] + val_analysis['total_sequences']
    
    comparison = {
        'total_sequences': total_sequences,
        'train_sequences': train_analysis['total_sequences'],
        'val_sequences': val_analysis['total_sequences'],
        'train_ratio': train_analysis['total_sequences'] / total_sequences,
        'val_ratio': val_analysis['total_sequences'] / total_sequences,
        
        'length_difference': {
            'train_mean': train_analysis['length_stats']['mean'],
            'val_mean': val_analysis['length_stats']['mean'],
            'difference': abs(train_analysis['length_stats']['mean'] - val_analysis['length_stats']['mean'])
        },
        
        'class_balance_difference': abs(
            train_analysis['class_balance'] - val_analysis['class_balance']
        ),
        
        'quality_difference': abs(
            train_analysis['quality_score'] - val_analysis['quality_score']
        )
    }
    
    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """
    Print dataset comparison results.
    
    Args:
        comparison: Comparison results from compare_datasets
    """
    print(f"\n{'='*60}")
    print("DATASET COMPARISON")
    print(f"{'='*60}")
    
    print(f"Total sequences:        {comparison['total_sequences']:,}")
    print(f"Training sequences:     {comparison['train_sequences']:,} ({comparison['train_ratio']*100:.1f}%)")
    print(f"Validation sequences:   {comparison['val_sequences']:,} ({comparison['val_ratio']*100:.1f}%)")
    
    print(f"\nLENGTH COMPARISON:")
    print(f"Training mean length:   {comparison['length_difference']['train_mean']:.1f}")
    print(f"Validation mean length: {comparison['length_difference']['val_mean']:.1f}")
    print(f"Length difference:      {comparison['length_difference']['difference']:.1f}")
    
    print(f"\nQUALITY COMPARISON:")
    print(f"Class balance diff:     {comparison['class_balance_difference']:.3f}")
    print(f"Quality score diff:     {comparison['quality_difference']:.3f}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if comparison['val_ratio'] < 0.1:
        print("⚠️  Validation set might be too small (< 10%)")
    elif comparison['val_ratio'] > 0.3:
        print("⚠️  Validation set might be too large (> 30%)")
    else:
        print("✅ Good train/validation split ratio")
    
    if comparison['length_difference']['difference'] > 2.0:
        print("⚠️  Significant length difference between train/val sets")
    else:
        print("✅ Similar length distributions")
    
    if comparison['class_balance_difference'] > 0.1:
        print("⚠️  Different class balance between train/val sets")
    else:
        print("✅ Similar class balance")
    
    print(f"{'='*60}")


def validate_twae_compatibility(analysis: Dict[str, Any], 
                              max_length: int = 37,
                              min_length: int = 3) -> List[str]:
    """
    Validate dataset compatibility with TWAE-MMD model.
    
    Args:
        analysis: Dataset analysis results
        max_length: Maximum sequence length for model
        min_length: Minimum sequence length for model
        
    Returns:
        List of compatibility issues (empty if no issues)
    """
    issues = []
    
    # Check sequence lengths
    if analysis['length_stats']['max'] > max_length - 1:  # -1 for special tokens
        issues.append(f"Some sequences too long (max: {analysis['length_stats']['max']}, limit: {max_length-1})")
    
    if analysis['length_stats']['min'] < min_length:
        issues.append(f"Some sequences too short (min: {analysis['length_stats']['min']}, limit: {min_length})")
    
    # Check for invalid characters
    if analysis['has_invalid_chars']:
        issues.append(f"Invalid amino acid characters found: {', '.join(analysis['invalid_characters'])}")
    
    # Check class balance
    if analysis['class_balance'] < 0.1:
        issues.append(f"Severe class imbalance (balance: {analysis['class_balance']:.3f})")
    
    # Check dataset size
    if analysis['total_sequences'] < 1000:
        issues.append(f"Dataset might be too small (size: {analysis['total_sequences']})")
    
    return issues

