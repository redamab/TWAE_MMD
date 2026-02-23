"""
Tokenization and Encoding for TWAE-MMD - Fixed and Simplified
============================================================

Simple, robust tokenization for antimicrobial peptide sequences.
This module provides essential tokenization functionality without complexity.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging


class PeptideTokenizer:
    """
    Simple, robust tokenizer for peptide sequences in TWAE-MMD.
    
    This tokenizer handles:
    - Amino acid vocabulary (20 standard amino acids)
    - Special tokens (PAD, UNK, CLS, SEP, MASK)
    - Sequence encoding/decoding
    - Padding and truncation
    - Attention mask generation
    """
    
    def __init__(self,
                 vocab_size: int = 25,
                 max_length: int = 37,
                 pad_token: str = "[PAD]",
                 unk_token: str = "[UNK]",
                 cls_token: str = "[CLS]",
                 sep_token: str = "[SEP]",
                 mask_token: str = "[MASK]"):
        """
        Initialize peptide tokenizer.
        
        Args:
            vocab_size: Vocabulary size (should be 25 for standard setup)
            max_length: Maximum sequence length
            pad_token: Padding token
            unk_token: Unknown token
            cls_token: Classification token
            sep_token: Separator token
            mask_token: Mask token for MLM
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        
        # Standard amino acids
        self.amino_acids = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
        ]
        
        # Build vocabulary
        self._build_vocabulary()
        
        # Special token IDs
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.cls_token_id = self.token_to_id[self.cls_token]
        self.sep_token_id = self.token_to_id[self.sep_token]
        self.mask_token_id = self.token_to_id[self.mask_token]
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
    
    def _build_vocabulary(self):
        """Build vocabulary with amino acids and special tokens."""
        # Start with special tokens
        vocab = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        
        # Add amino acids
        vocab.extend(self.amino_acids)
        
        # Ensure vocabulary size matches
        if len(vocab) != self.vocab_size:
            raise ValueError(f"Vocabulary size mismatch: expected {self.vocab_size}, got {len(vocab)}")
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab = vocab
    
    def encode(self,
               sequence: str,
               add_special_tokens: bool = True,
               padding: str = 'max_length',
               truncation: bool = True,
               return_attention_mask: bool = True) -> Dict[str, List[int]]:
        """
        Encode a peptide sequence to token IDs.
        
        Args:
            sequence: Peptide sequence string
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            padding: Padding strategy ('max_length' or 'do_not_pad')
            truncation: Whether to truncate sequences
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        # Clean sequence
        sequence = sequence.upper().strip()
        
        # Convert to token IDs
        token_ids = []
        
        # Add CLS token if requested
        if add_special_tokens:
            token_ids.append(self.cls_token_id)
        
        # Add sequence tokens
        for char in sequence:
            if char in self.token_to_id:
                token_ids.append(self.token_to_id[char])
            else:
                token_ids.append(self.unk_token_id)
        
        # Add SEP token if requested
        if add_special_tokens:
            token_ids.append(self.sep_token_id)
        
        # Truncate if necessary
        if truncation and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            # Ensure SEP token is at the end if special tokens are used
            if add_special_tokens:
                token_ids[-1] = self.sep_token_id
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad if necessary
        if padding == 'max_length':
            while len(token_ids) < self.max_length:
                token_ids.append(self.pad_token_id)
                attention_mask.append(0)
        
        # Prepare result
        result = {'input_ids': token_ids}
        if return_attention_mask:
            result['attention_mask'] = attention_mask
        
        return result
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to sequence string.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded sequence string
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in [self.pad_token, self.cls_token, 
                                                   self.sep_token, self.mask_token]:
                    continue
                
                # Skip unknown tokens if requested
                if skip_special_tokens and token == self.unk_token:
                    continue
                
                tokens.append(token)
        
        return ''.join(tokens)
    
    def batch_encode(self,
                     sequences: List[str],
                     add_special_tokens: bool = True,
                     padding: str = 'max_length',
                     truncation: bool = True,
                     return_attention_mask: bool = True) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of peptide sequences.
        
        Args:
            sequences: List of peptide sequence strings
            add_special_tokens: Whether to add [CLS] and [SEP] tokens
            padding: Padding strategy
            truncation: Whether to truncate sequences
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with batched 'input_ids' and optionally 'attention_mask'
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        for sequence in sequences:
            encoded = self.encode(
                sequence=sequence,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                return_attention_mask=return_attention_mask
            )
            
            batch_input_ids.append(encoded['input_ids'])
            if return_attention_mask:
                batch_attention_mask.append(encoded['attention_mask'])
        
        result = {'input_ids': batch_input_ids}
        if return_attention_mask:
            result['attention_mask'] = batch_attention_mask
        
        return result
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self.token_to_id.copy()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return [self.id_to_token.get(id, self.unk_token) for id in ids]
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file."""
        import json
        vocab_data = {
            'token_to_id': self.token_to_id,
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'cls_token': self.cls_token,
                'sep_token': self.sep_token,
                'mask_token': self.mask_token
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    @classmethod
    def load_vocabulary(cls, filepath: str) -> 'PeptideTokenizer':
        """Load vocabulary from file."""
        import json
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(
            vocab_size=vocab_data['vocab_size'],
            max_length=vocab_data['max_length'],
            **vocab_data['special_tokens']
        )
        
        return tokenizer


def create_tokenizer(vocab_size: int = 25,
                    max_length: int = 37,
                    **kwargs) -> PeptideTokenizer:
    """
    Create a peptide tokenizer with default settings.
    
    Args:
        vocab_size: Vocabulary size
        max_length: Maximum sequence length
        **kwargs: Additional tokenizer arguments
        
    Returns:
        PeptideTokenizer instance
    """
    return PeptideTokenizer(
        vocab_size=vocab_size,
        max_length=max_length,
        **kwargs
    )

