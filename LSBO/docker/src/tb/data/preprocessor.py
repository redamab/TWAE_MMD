"""
Data Preprocessing Pipeline for TWAE-MMD - Fixed and Simplified
==============================================================

Simple, robust preprocessing for antimicrobial peptide sequences.
This module provides essential preprocessing functionality without complexity.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Tuple, Dict, Optional, Set
from collections import Counter


class PeptidePreprocessor:
    """
    Simple, robust peptide sequence preprocessor for TWAE-MMD.
    
    This preprocessor handles:
    - Sequence cleaning and validation
    - Length filtering (3-36 amino acids)
    - Invalid character removal
    - Duplicate detection and removal
    - Quality control and statistics
    """
    
    def __init__(self,
                 min_length: int = 3,
                 max_length: int = 36,
                 valid_amino_acids: Optional[Set[str]] = None,
                 remove_duplicates: bool = True,
                 case_sensitive: bool = False,
                 strict_validation: bool = True):
        """
        Initialize peptide preprocessor.
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            valid_amino_acids: Set of valid amino acids (default: standard 20)
            remove_duplicates: Whether to remove duplicate sequences
            case_sensitive: Whether to treat sequences as case-sensitive
            strict_validation: Whether to use strict validation
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.case_sensitive = case_sensitive
        self.strict_validation = strict_validation
        
        # Set valid amino acids
        if valid_amino_acids is None:
            self.valid_amino_acids = {
                'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
            }
        else:
            self.valid_amino_acids = set(valid_amino_acids)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'total_sequences': 0,
            'cleaned_sequences': 0,
            'valid_sequences': 0,
            'final_sequences': 0,
            'removed_duplicates': 0,
            'removed_invalid': 0,
            'removed_length': 0
        }
    
    def clean_sequence(self, sequence: str) -> str:
        """
        Clean a single peptide sequence.
        
        Args:
            sequence: Raw peptide sequence
            
        Returns:
            Cleaned sequence
        """
        if not isinstance(sequence, str):
            return ""
        
        # Remove whitespace and convert to uppercase
        cleaned = sequence.strip().upper()
        
        # Remove non-amino acid characters
        cleaned = re.sub(r'[^ARNDCQEGHILKMFPSTWYV]', '', cleaned)
        
        return cleaned
    
    def validate_sequence(self, sequence: str) -> bool:
        """
        Validate a peptide sequence.
        
        Args:
            sequence: Peptide sequence to validate
            
        Returns:
            True if sequence is valid, False otherwise
        """
        # Check if sequence is empty
        if not sequence:
            return False
        
        # Check length
        if len(sequence) < self.min_length or len(sequence) > self.max_length:
            return False
        
        # Check for valid amino acids
        if self.strict_validation:
            for char in sequence:
                if char not in self.valid_amino_acids:
                    return False
        
        return True
    
    def compute_sequence_properties(self, sequence: str) -> Dict[str, float]:
        """
        Compute basic properties of a peptide sequence.
        
        Args:
            sequence: Peptide sequence
            
        Returns:
            Dictionary of sequence properties
        """
        if not sequence:
            return {'length': 0, 'hydrophobicity': 0.0, 'charge': 0.0}
        
        # Hydrophobicity scale (Kyte-Doolittle)
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        # Charge scale
        charge_scale = {
            'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1
        }
        
        # Calculate properties
        length = len(sequence)
        hydrophobicity = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / length
        charge = sum(charge_scale.get(aa, 0) for aa in sequence)
        
        return {
            'length': length,
            'hydrophobicity': hydrophobicity,
            'charge': charge
        }
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           sequence_column: str = 'sequence',
                           label_column: str = 'label') -> pd.DataFrame:
        """
        Preprocess a DataFrame of peptide sequences.
        
        Args:
            df: Input DataFrame
            sequence_column: Name of sequence column
            label_column: Name of label column
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info(f"Starting preprocessing of {len(df)} sequences...")
        
        # Reset statistics
        self.stats['total_sequences'] = len(df)
        
        # Make a copy
        processed_df = df.copy()
        
        # Clean sequences
        self.logger.info("Cleaning sequences...")
        processed_df[sequence_column] = processed_df[sequence_column].apply(self.clean_sequence)
        self.stats['cleaned_sequences'] = len(processed_df)
        
        # Validate sequences
        self.logger.info("Validating sequences...")
        valid_mask = processed_df[sequence_column].apply(self.validate_sequence)
        processed_df = processed_df[valid_mask].copy()
        self.stats['valid_sequences'] = len(processed_df)
        self.stats['removed_invalid'] = self.stats['cleaned_sequences'] - self.stats['valid_sequences']
        
        # Remove duplicates if requested
        if self.remove_duplicates:
            self.logger.info("Removing duplicate sequences...")
            initial_count = len(processed_df)
            processed_df = processed_df.drop_duplicates(subset=[sequence_column]).copy()
            self.stats['removed_duplicates'] = initial_count - len(processed_df)
        
        # Compute sequence properties
        self.logger.info("Computing sequence properties...")
        properties = processed_df[sequence_column].apply(self.compute_sequence_properties)
        
        # Add properties as columns
        processed_df['length'] = [p['length'] for p in properties]
        processed_df['hydrophobicity'] = [p['hydrophobicity'] for p in properties]
        processed_df['charge'] = [p['charge'] for p in properties]
        
        self.stats['final_sequences'] = len(processed_df)
        
        self.logger.info(f"Preprocessing complete. {self.stats['final_sequences']} sequences remaining.")
        
        return processed_df
    
    def get_statistics(self) -> Dict[str, int]:
        """Get preprocessing statistics."""
        return self.stats.copy()
    
    def print_statistics(self):
        """Print preprocessing statistics."""
        print("\n" + "="*50)
        print("PREPROCESSING STATISTICS")
        print("="*50)
        print(f"Total sequences:        {self.stats['total_sequences']:,}")
        print(f"After cleaning:         {self.stats['cleaned_sequences']:,}")
        print(f"After validation:       {self.stats['valid_sequences']:,}")
        print(f"Final sequences:        {self.stats['final_sequences']:,}")
        print(f"Removed (invalid):      {self.stats['removed_invalid']:,}")
        print(f"Removed (duplicates):   {self.stats['removed_duplicates']:,}")
        print(f"Removal rate:           {(1 - self.stats['final_sequences']/self.stats['total_sequences'])*100:.1f}%")
        print("="*50)
    
    def analyze_sequences(self, df: pd.DataFrame, sequence_column: str = 'sequence') -> Dict[str, any]:
        """
        Analyze sequence characteristics.
        
        Args:
            df: DataFrame with sequences
            sequence_column: Name of sequence column
            
        Returns:
            Analysis results
        """
        sequences = df[sequence_column].tolist()
        
        # Length distribution
        lengths = [len(seq) for seq in sequences]
        
        # Amino acid composition
        all_aas = ''.join(sequences)
        aa_counts = Counter(all_aas)
        
        # Analysis results
        analysis = {
            'total_sequences': len(sequences),
            'length_stats': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'median': np.median(lengths),
                'std': np.std(lengths)
            },
            'amino_acid_composition': dict(aa_counts),
            'total_amino_acids': len(all_aas),
            'unique_amino_acids': len(set(all_aas))
        }
        
        return analysis


def create_preprocessor(min_length: int = 3,
                       max_length: int = 36,
                       **kwargs) -> PeptidePreprocessor:
    """
    Create a peptide preprocessor with default settings.
    
    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        **kwargs: Additional preprocessor arguments
        
    Returns:
        PeptidePreprocessor instance
    """
    return PeptidePreprocessor(
        min_length=min_length,
        max_length=max_length,
        **kwargs
    )

