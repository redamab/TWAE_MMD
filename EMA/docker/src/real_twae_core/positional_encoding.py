"""
Positional Encoding for TWAE-MMD Transformer
Optimized for antimicrobial peptide sequences (3-36 amino acids)

This module implements both sinusoidal and learned positional encodings
specifically designed for peptide sequence modeling in the TWAE-MMD architecture.
"""

import tensorflow as tf
import numpy as np
import math
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(tf.keras.layers.Layer):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need".
    
    This implementation is optimized for peptide sequences with lengths up to 37 tokens.
    The sinusoidal encoding provides good generalization to unseen sequence lengths.
    """
    
    def __init__(self, 
                 max_length: int = 37,
                 d_model: int = 256,
                 temperature: float = 10000.0,
                 name: str = "sinusoidal_pos_encoding"):
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            max_length: Maximum sequence length
            d_model: Model dimension
            temperature: Temperature parameter for frequency scaling
            name: Layer name
        """
        super().__init__(name=name)
        
        self.max_length = max_length
        self.d_model = d_model
        self.temperature = temperature
        
        # Pre-compute positional encodings
        self.pos_encoding = self._create_positional_encoding()
    
    def _create_positional_encoding(self) -> tf.Tensor:
        """
        Create sinusoidal positional encoding matrix.
        
        Returns:
            Positional encoding tensor [max_length, d_model]
        """
        # Create position indices
        position = tf.range(self.max_length, dtype=tf.float32)[:, tf.newaxis]
        
        # Create dimension indices
        div_term = tf.exp(
            tf.range(0, self.d_model, 2, dtype=tf.float32) * 
            -(math.log(self.temperature) / self.d_model)
        )
        
        # Initialize positional encoding matrix
        pos_encoding = tf.zeros((self.max_length, self.d_model))
        
        # Apply sine to even indices
        pos_encoding_even = tf.sin(position * div_term)
        
        # Apply cosine to odd indices
        pos_encoding_odd = tf.cos(position * div_term)
        
        # Interleave sine and cosine
        pos_encoding = tf.stack([pos_encoding_even, pos_encoding_odd], axis=2)
        pos_encoding = tf.reshape(pos_encoding, (self.max_length, self.d_model))
        
        # Handle odd d_model
        if self.d_model % 2 == 1:
            pos_encoding = pos_encoding[:, :-1]
            # Add an extra dimension with zeros
            extra_dim = tf.zeros((self.max_length, 1))
            pos_encoding = tf.concat([pos_encoding, extra_dim], axis=1)
        
        return pos_encoding
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Apply positional encoding to input embeddings.
        
        Args:
            inputs: Input embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        """
        seq_len = tf.shape(inputs)[1]
        
        # Get positional encodings for the sequence length
        pos_encoding = self.pos_encoding[:seq_len, :]
        
        # Add positional encoding to inputs
        return inputs + pos_encoding
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'max_length': self.max_length,
            'd_model': self.d_model,
            'temperature': self.temperature,
        })
        return config


class LearnedPositionalEncoding(tf.keras.layers.Layer):
    """
    Learned positional encoding using trainable embeddings.
    
    This approach learns position-specific representations that may be more
    suitable for peptide sequences with specific structural patterns.
    """
    
    def __init__(self,
                 max_length: int = 37,
                 d_model: int = 256,
                 dropout_rate: float = 0.1,
                 name: str = "learned_pos_encoding"):
        """
        Initialize learned positional encoding.
        
        Args:
            max_length: Maximum sequence length
            d_model: Model dimension
            dropout_rate: Dropout rate for regularization
            name: Layer name
        """
        super().__init__(name=name)
        
        self.max_length = max_length
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        
        # Learnable positional embeddings
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=max_length,
            output_dim=d_model,
            embeddings_initializer='uniform',
            name='position_embeddings'
        )
        
        # Dropout for regularization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply learned positional encoding to input embeddings.
        
        Args:
            inputs: Input embeddings [batch_size, seq_len, d_model]
            training: Training mode
            
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Create position indices
        positions = tf.range(seq_len)
        positions = tf.expand_dims(positions, 0)
        positions = tf.tile(positions, [batch_size, 1])
        
        # Get positional embeddings
        pos_embeddings = self.pos_embedding(positions)
        
        # Add positional encoding to inputs
        outputs = inputs + pos_embeddings
        
        # Apply dropout
        outputs = self.dropout(outputs, training=training)
        
        return outputs
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'max_length': self.max_length,
            'd_model': self.d_model,
            'dropout_rate': self.dropout_rate,
        })
        return config


class RelativePositionalEncoding(tf.keras.layers.Layer):
    """
    Relative positional encoding for better handling of variable-length sequences.
    
    This encoding focuses on relative distances between positions rather than
    absolute positions, which can be beneficial for peptide sequences.
    """
    
    def __init__(self,
                 max_relative_distance: int = 32,
                 d_model: int = 256,
                 num_heads: int = 8,
                 name: str = "relative_pos_encoding"):
        """
        Initialize relative positional encoding.
        
        Args:
            max_relative_distance: Maximum relative distance to consider
            d_model: Model dimension
            num_heads: Number of attention heads
            name: Layer name
        """
        super().__init__(name=name)
        
        self.max_relative_distance = max_relative_distance
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Relative position embeddings
        self.relative_pos_embedding = tf.keras.layers.Embedding(
            input_dim=2 * max_relative_distance + 1,
            output_dim=self.head_dim,
            embeddings_initializer='glorot_uniform',
            name='relative_position_embeddings'
        )
    
    def _get_relative_positions(self, seq_len: int) -> tf.Tensor:
        """
        Get relative position matrix.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position matrix [seq_len, seq_len]
        """
        # Create position indices
        positions = tf.range(seq_len)
        
        # Compute relative positions
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to maximum relative distance
        relative_positions = tf.clip_by_value(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance
        )
        
        # Shift to make all values positive
        relative_positions += self.max_relative_distance
        
        return relative_positions
    
    def call(self, seq_len: int) -> tf.Tensor:
        """
        Get relative positional encoding for given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative positional encoding [seq_len, seq_len, head_dim]
        """
        # Get relative position matrix
        relative_positions = self._get_relative_positions(seq_len)
        
        # Get relative position embeddings
        relative_pos_embeddings = self.relative_pos_embedding(relative_positions)
        
        return relative_pos_embeddings
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'max_relative_distance': self.max_relative_distance,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
        })
        return config


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Unified positional encoding layer that supports multiple encoding types.
    
    This layer provides a flexible interface for different positional encoding
    strategies optimized for peptide sequence modeling.
    """
    
    def __init__(self,
                 max_length: int = 37,
                 d_model: int = 256,
                 encoding_type: str = "learned",
                 dropout_rate: float = 0.1,
                 temperature: float = 10000.0,
                 name: str = "positional_encoding"):
        """
        Initialize positional encoding layer.
        
        Args:
            max_length: Maximum sequence length
            d_model: Model dimension
            encoding_type: Type of encoding ('sinusoidal', 'learned', 'relative')
            dropout_rate: Dropout rate for regularization
            temperature: Temperature for sinusoidal encoding
            name: Layer name
        """
        super().__init__(name=name)
        
        self.max_length = max_length
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        
        # Initialize the appropriate encoding layer
        if encoding_type == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(
                max_length=max_length,
                d_model=d_model,
                temperature=temperature,
                name=f"{name}_sinusoidal"
            )
        elif encoding_type == "learned":
            self.pos_encoder = LearnedPositionalEncoding(
                max_length=max_length,
                d_model=d_model,
                dropout_rate=dropout_rate,
                name=f"{name}_learned"
            )
        elif encoding_type == "relative":
            self.pos_encoder = RelativePositionalEncoding(
                max_relative_distance=min(32, max_length),
                d_model=d_model,
                name=f"{name}_relative"
            )
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        # Additional dropout layer
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply positional encoding to input embeddings.
        
        Args:
            inputs: Input embeddings [batch_size, seq_len, d_model]
            training: Training mode
            
        Returns:
            Embeddings with positional encoding [batch_size, seq_len, d_model]
        """
        if self.encoding_type == "relative":
            # Relative encoding is handled differently in attention layers
            return self.dropout(inputs, training=training)
        else:
            # Apply positional encoding
            outputs = self.pos_encoder(inputs, training=training)
            return self.dropout(outputs, training=training)
    
    def get_relative_encoding(self, seq_len: int) -> Optional[tf.Tensor]:
        """
        Get relative positional encoding if using relative encoding.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative positional encoding or None
        """
        if self.encoding_type == "relative":
            return self.pos_encoder(seq_len)
        return None
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'max_length': self.max_length,
            'd_model': self.d_model,
            'encoding_type': self.encoding_type,
            'dropout_rate': self.dropout_rate,
            'temperature': self.temperature,
        })
        return config


def create_positional_encoding(encoding_type: str = "learned",
                              max_length: int = 37,
                              d_model: int = 256,
                              dropout_rate: float = 0.1,
                              name: str = "positional_encoding") -> PositionalEncoding:
    """
    Create positional encoding layer.
    
    Args:
        encoding_type: Type of encoding ("sinusoidal" or "learned")
        max_length: Maximum sequence length
        d_model: Model dimension
        dropout_rate: Dropout rate
        name: Layer name
        
    Returns:
        PositionalEncoding layer
    """
    return PositionalEncoding(
        max_length=max_length,
        d_model=d_model,
        encoding_type=encoding_type,
        dropout_rate=dropout_rate,
        name=name
    )


# Utility functions for positional encoding analysis
def visualize_sinusoidal_encoding(max_length: int = 37, 
                                 d_model: int = 256,
                                 save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize sinusoidal positional encoding patterns.
    
    Args:
        max_length: Maximum sequence length
        d_model: Model dimension
        save_path: Path to save visualization (optional)
        
    Returns:
        Positional encoding matrix
    """
    pos_encoder = SinusoidalPositionalEncoding(max_length, d_model)
    
    # Extract positional encoding directly from the encoder
    pos_encoding = pos_encoder.positional_encoding[:max_length, :].numpy()
    
    if save_path:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        plt.imshow(pos_encoding.T, aspect='auto', cmap='RdBu')
        plt.colorbar()
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title('Sinusoidal Positional Encoding')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return pos_encoding


