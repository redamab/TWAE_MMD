"""
Biological Constraints for AMP Generation
==========================================

This module provides constraint-based filtering for antimicrobial peptides
to ensure biological validity and drug-like properties.

Constraints include:
- Length (10-30 amino acids)
- Charge (+2 to +9 for AMPs)
- Hydrophobicity (30-70%)
- Required/forbidden amino acids
- Sequence diversity

"""

import tensorflow as tf
import numpy as np
from typing import List, Set, Optional, Union


class AMPConstraints:
    """
    Biological constraints for antimicrobial peptides.
    
    AMPs typically have specific properties:
    - Length: 10-30 amino acids
    - Charge: +2 to +9 (cationic)
    - Hydrophobicity: 30-70%
    - Contains lysine (K) or arginine (R)
    - Avoids cysteines (C) for stability
    """
    
    def __init__(self,
                 min_length: int = 10,
                 max_length: int = 36,
                 min_charge: float = 2.0,
                 max_charge: float = 9.0,
                 min_hydrophobicity: float = 0.3,
                 max_hydrophobicity: float = 0.7,
                 required_amino_acids: Optional[Set[str]] = None,
                 forbidden_amino_acids: Optional[Set[str]] = None,
                 min_diversity: float = 0.3,
                 tokenizer = None):
        """
        Initialize AMP constraints.
        
        Args:
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            min_charge: Minimum net charge
            max_charge: Maximum net charge
            min_hydrophobicity: Minimum hydrophobicity ratio
            max_hydrophobicity: Maximum hydrophobicity ratio
            required_amino_acids: Set of required amino acids
            forbidden_amino_acids: Set of forbidden amino acids
            min_diversity: Minimum amino acid diversity
            tokenizer: Tokenizer to decode sequences
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_charge = min_charge
        self.max_charge = max_charge
        self.min_hydrophobicity = min_hydrophobicity
        self.max_hydrophobicity = max_hydrophobicity
        self.required_amino_acids = required_amino_acids or {'K', 'R'}  # Lysine or Arginine
        self.forbidden_amino_acids = forbidden_amino_acids or set()
        self.min_diversity = min_diversity
        self.tokenizer = tokenizer
        
        # Amino acid properties
        self.charged_positive = {'K', 'R', 'H'}  # Lysine, Arginine, Histidine
        self.charged_negative = {'D', 'E'}  # Aspartate, Glutamate
        self.hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'}
        self.polar = {'S', 'T', 'N', 'Q', 'Y', 'C'}
    
    def satisfies(self, sequence: Union[tf.Tensor, str]) -> bool:
        """
        Check if sequence satisfies all constraints.
        
        Args:
            sequence: Sequence tensor [seq_len] or string
            
        Returns:
            True if all constraints satisfied, False otherwise
        """
        # Convert to string
        if isinstance(sequence, tf.Tensor):
            seq_str = self._decode_sequence(sequence)
        else:
            seq_str = sequence
        
        # Remove padding and special tokens
        seq_str = seq_str.replace('<PAD>', '').replace('<START>', '').replace('<END>', '').strip()
        
        # Check length
        if not (self.min_length <= len(seq_str) <= self.max_length):
            return False
        
        # Check charge
        charge = self._calculate_charge(seq_str)
        if not (self.min_charge <= charge <= self.max_charge):
            return False
        
        # Check hydrophobicity
        hydrophobicity = self._calculate_hydrophobicity(seq_str)
        if not (self.min_hydrophobicity <= hydrophobicity <= self.max_hydrophobicity):
            return False
        
        # Check required amino acids
        if self.required_amino_acids:
            if not any(aa in seq_str for aa in self.required_amino_acids):
                return False
        
        # Check forbidden amino acids
        if self.forbidden_amino_acids:
            if any(aa in seq_str for aa in self.forbidden_amino_acids):
                return False
        
        # Check diversity
        diversity = self._calculate_diversity(seq_str)
        if diversity < self.min_diversity:
            return False
        
        return True
    
    def filter_sequences(self, sequences: Union[tf.Tensor, List[str]]) -> List:
        """
        Filter sequences by constraints.
        
        Args:
            sequences: List of sequences (tensors or strings)
            
        Returns:
            List of valid sequences
        """
        valid_sequences = []
        for seq in sequences:
            if self.satisfies(seq):
                valid_sequences.append(seq)
        return valid_sequences
    
    def get_violation_reasons(self, sequence: Union[tf.Tensor, str]) -> List[str]:
        """
        Get reasons why sequence violates constraints.
        
        Args:
            sequence: Sequence tensor or string
            
        Returns:
            List of violation reasons
        """
        reasons = []
        
        # Convert to string
        if isinstance(sequence, tf.Tensor):
            seq_str = self._decode_sequence(sequence)
        else:
            seq_str = sequence
        
        seq_str = seq_str.replace('<PAD>', '').replace('<START>', '').replace('<END>', '').strip()
        
        # Check length
        if not (self.min_length <= len(seq_str) <= self.max_length):
            reasons.append(f"Length {len(seq_str)} not in [{self.min_length}, {self.max_length}]")
        
        # Check charge
        charge = self._calculate_charge(seq_str)
        if not (self.min_charge <= charge <= self.max_charge):
            reasons.append(f"Charge {charge:.1f} not in [{self.min_charge}, {self.max_charge}]")
        
        # Check hydrophobicity
        hydrophobicity = self._calculate_hydrophobicity(seq_str)
        if not (self.min_hydrophobicity <= hydrophobicity <= self.max_hydrophobicity):
            reasons.append(f"Hydrophobicity {hydrophobicity:.2f} not in [{self.min_hydrophobicity}, {self.max_hydrophobicity}]")
        
        # Check required amino acids
        if self.required_amino_acids:
            if not any(aa in seq_str for aa in self.required_amino_acids):
                reasons.append(f"Missing required amino acids: {self.required_amino_acids}")
        
        # Check forbidden amino acids
        if self.forbidden_amino_acids:
            forbidden_found = [aa for aa in self.forbidden_amino_acids if aa in seq_str]
            if forbidden_found:
                reasons.append(f"Contains forbidden amino acids: {forbidden_found}")
        
        # Check diversity
        diversity = self._calculate_diversity(seq_str)
        if diversity < self.min_diversity:
            reasons.append(f"Diversity {diversity:.2f} < {self.min_diversity}")
        
        return reasons
    
    def _decode_sequence(self, sequence: tf.Tensor) -> str:
        """Decode sequence tensor to string."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(sequence.numpy())
        else:
            # Simple decoding (assumes tokens are ASCII-like)
            return ''.join([chr(x + 65) for x in sequence.numpy() if x > 0])
    
    def _calculate_charge(self, sequence: str) -> float:
        """Calculate net charge of sequence."""
        positive = sum(1 for aa in sequence if aa in self.charged_positive)
        negative = sum(1 for aa in sequence if aa in self.charged_negative)
        return positive - negative
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """Calculate hydrophobicity ratio."""
        if len(sequence) == 0:
            return 0.0
        hydrophobic_count = sum(1 for aa in sequence if aa in self.hydrophobic)
        return hydrophobic_count / len(sequence)
    
    def _calculate_diversity(self, sequence: str) -> float:
        """Calculate amino acid diversity."""
        if len(sequence) == 0:
            return 0.0
        unique_aa = len(set(sequence))
        return unique_aa / len(sequence)
    
    def __repr__(self) -> str:
        """String representation of constraints."""
        return f"""AMPConstraints(
    length=[{self.min_length}, {self.max_length}],
    charge=[{self.min_charge}, {self.max_charge}],
    hydrophobicity=[{self.min_hydrophobicity}, {self.max_hydrophobicity}],
    required_aa={self.required_amino_acids},
    forbidden_aa={self.forbidden_amino_acids},
    min_diversity={self.min_diversity}
)"""


class ConstrainedSampler:
    """
    Constrained sampler that generates valid AMPs.
    
    Combines random sampling with constraint filtering to ensure
    all generated sequences are biologically valid.
    """
    
    def __init__(self, model, constraints: AMPConstraints):
        """
        Initialize constrained sampler.
        
        Args:
            model: Trained TWAE model
            constraints: AMP constraints
        """
        self.model = model
        self.constraints = constraints
    
    def sample(self,
               num_samples: int = 100,
               temperature: float = 1.0,
               max_attempts: int = 1000,
               verbose: bool = False) -> tf.Tensor:
        """
        Sample valid AMPs with constraints.
        
        Args:
            num_samples: Number of valid samples to generate
            temperature: Sampling temperature
            max_attempts: Maximum attempts before giving up
            verbose: Print progress
            
        Returns:
            Valid sequences [num_samples, seq_len]
        """
        valid_sequences = []
        attempts = 0
        
        if verbose:
            print(f"Generating {num_samples} valid AMPs with constraints...")
            print(f"Constraints: {self.constraints}")
        
        while len(valid_sequences) < num_samples and attempts < max_attempts:
            # Generate batch
            batch_size = min(32, num_samples - len(valid_sequences) + 10)
            batch = self.model.generate_sequences(
                num_samples=batch_size,
                temperature=temperature,
                sampling_strategy='gaussian'
            )
            
            # Filter by constraints
            for seq in batch:
                if self.constraints.satisfies(seq):
                    valid_sequences.append(seq)
                    if len(valid_sequences) >= num_samples:
                        break
            
            attempts += batch_size
            
            if verbose and len(valid_sequences) % 10 == 0:
                success_rate = len(valid_sequences) / attempts * 100
                print(f"  Generated {len(valid_sequences)}/{num_samples} "
                      f"(success rate: {success_rate:.1f}%)")
        
        if len(valid_sequences) < num_samples:
            print(f"⚠️ Warning: Only generated {len(valid_sequences)}/{num_samples} "
                  f"valid sequences after {attempts} attempts")
        
        # Convert to tensor
        valid_tensor = tf.stack(valid_sequences[:num_samples])
        
        if verbose:
            print(f" Generated {len(valid_tensor)} valid AMPs!")
        
        return valid_tensor


