"""
Multi-Head Attention Layers for TWAE-MMD Transformer
Optimized for antimicrobial peptide sequence modeling

This module implements efficient multi-head attention mechanisms with
peptide-specific optimizations for the TWAE-MMD architecture.
"""

import tensorflow as tf
import numpy as np
import math
from typing import Optional, Tuple, Union


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer optimized for peptide sequences.
    
    This implementation includes several optimizations:
    - Efficient attention computation with optional relative positioning
    - Attention dropout for regularization
    - Support for causal and bidirectional attention
    - Memory-efficient implementation for long sequences
    """
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_bias: bool = True,
                 use_relative_position: bool = False,
                 max_relative_distance: int = 32,
                 causal: bool = False,
                 name: str = "multi_head_attention"):
        """
        Initialize multi-head attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for output
            attention_dropout: Dropout rate for attention weights
            use_bias: Whether to use bias in linear projections
            use_relative_position: Whether to use relative positional encoding
            max_relative_distance: Maximum relative distance for relative encoding
            causal: Whether to use causal (autoregressive) attention
            name: Layer name
        """
        super().__init__(name=name)
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.use_relative_position = use_relative_position
        self.max_relative_distance = max_relative_distance
        self.causal = causal
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.query_projection = tf.keras.layers.Dense(
            d_model, use_bias=use_bias, name="query_projection"
        )
        self.key_projection = tf.keras.layers.Dense(
            d_model, use_bias=use_bias, name="key_projection"
        )
        self.value_projection = tf.keras.layers.Dense(
            d_model, use_bias=use_bias, name="value_projection"
        )
        
        # Output projection
        self.output_projection = tf.keras.layers.Dense(
            d_model, use_bias=use_bias, name="output_projection"
        )
        
        # Dropout layers
        self.attention_dropout_layer = tf.keras.layers.Dropout(attention_dropout)
        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Relative position encoding (if enabled)
        if use_relative_position:
            self.relative_pos_embedding = tf.keras.layers.Embedding(
                input_dim=2 * max_relative_distance + 1,
                output_dim=self.head_dim,
                embeddings_initializer='glorot_uniform',
                name='relative_position_embeddings'
            )
    
    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """
        Split the last dimension into (num_heads, head_dim).
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Reshaped tensor [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        """
        Merge attention heads back to original dimension.
        
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Merged tensor [batch_size, seq_len, d_model]
        """
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[2]
        
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, seq_len, self.d_model))
    
    def _get_relative_positions(self, seq_len: int) -> tf.Tensor:
        """
        Get relative position matrix for relative positional encoding.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position matrix [seq_len, seq_len]
        """
        positions = tf.range(seq_len)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to maximum relative distance
        relative_positions = tf.clip_by_value(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance
        )
        
        # Shift to make all values positive
        return relative_positions + self.max_relative_distance
    
    def _compute_attention_scores(self,
                                 query: tf.Tensor,
                                 key: tf.Tensor,
                                 seq_len: int) -> tf.Tensor:
        """
        Compute attention scores with optional relative positioning.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            seq_len: Sequence length
            
        Returns:
            Attention scores [batch_size, num_heads, seq_len, seq_len]
        """
        # Standard attention scores
        scores = tf.matmul(query, key, transpose_b=True) * self.scale
        
        # Add relative positional encoding if enabled
        if self.use_relative_position:
            relative_positions = self._get_relative_positions(seq_len)
            relative_embeddings = self.relative_pos_embedding(relative_positions)
            
            # Compute relative attention scores
            # query: [batch_size, num_heads, seq_len, head_dim]
            # relative_embeddings: [seq_len, seq_len, head_dim]
            relative_scores = tf.einsum('bhid,jkd->bhijk', query, relative_embeddings)
            relative_scores = tf.reduce_sum(relative_scores, axis=-1)  # Sum over head_dim
            
            scores += relative_scores
        
        return scores
    
    def _apply_attention_mask(self,
                             scores: tf.Tensor,
                             mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply attention mask to scores with proper mixed precision support.
        
        Args:
            scores: Attention scores [batch_size, num_heads, seq_len, seq_len]
            mask: Attention mask [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
            
        Returns:
            Masked attention scores
        """
        if mask is not None:
            # Convert mask to attention mask (0 for valid, -inf for invalid)
            # FIXED: Cast mask to same dtype as scores for mixed precision compatibility
            mask = tf.cast(mask, scores.dtype)
            mask = (1.0 - mask) * -1e9
            scores += mask
        
        # Apply causal mask if needed
        if self.causal:
            seq_len = tf.shape(scores)[-1]
            causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            causal_mask = tf.where(causal_mask == 0, -1e9, 0.0)
            # FIXED: Cast causal mask to same dtype as scores for mixed precision compatibility
            causal_mask = tf.cast(causal_mask, scores.dtype)
            scores += causal_mask
        
        return scores
    
    def call(self,
             query: tf.Tensor,
             key: Optional[tf.Tensor] = None,
             value: Optional[tf.Tensor] = None,
             mask: Optional[tf.Tensor] = None,
             training: bool = False,
             return_attention_weights: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Apply multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model] (defaults to query)
            value: Value tensor [batch_size, seq_len, d_model] (defaults to key)
            mask: Attention mask [batch_size, 1, 1, seq_len]
            training: Training mode
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor [batch_size, seq_len, d_model] and optionally attention weights
        """
        # Default key and value to query (self-attention)
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        
        # Linear projections
        Q = self.query_projection(query)  # [batch_size, seq_len, d_model]
        K = self.key_projection(key)      # [batch_size, seq_len, d_model]
        V = self.value_projection(value)  # [batch_size, seq_len, d_model]
        
        # Split into multiple heads
        Q = self._split_heads(Q)  # [batch_size, num_heads, seq_len, head_dim]
        K = self._split_heads(K)  # [batch_size, num_heads, seq_len, head_dim]
        V = self._split_heads(V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        scores = self._compute_attention_scores(Q, K, seq_len)
        
        # Apply attention mask
        scores = self._apply_attention_mask(scores, mask)
        
        # Compute attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.attention_dropout_layer(attention_weights, training=training)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, V)
        
        # Merge heads
        attention_output = self._merge_heads(attention_output)
        
        # Final output projection
        output = self.output_projection(attention_output)
        output = self.output_dropout(output, training=training)
        
        if return_attention_weights:
            return output, attention_weights
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'use_bias': self.use_bias,
            'use_relative_position': self.use_relative_position,
            'max_relative_distance': self.max_relative_distance,
            'causal': self.causal,
        })
        return config


class CrossAttention(tf.keras.layers.Layer):
    """
    Cross-attention layer for encoder-decoder architectures.
    
    This layer performs attention between different sequences, useful for
    connecting the encoder and decoder in the TWAE-MMD architecture.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_bias: bool = True,
                 name: str = "cross_attention"):
        """
        Initialize cross-attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for output
            attention_dropout: Dropout rate for attention weights
            use_bias: Whether to use bias in linear projections
            name: Layer name
        """
        super().__init__(name=name)
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            use_bias=use_bias,
            use_relative_position=False,  # Usually not used in cross-attention
            causal=False,
            name=f"{name}_mha"
        )
    
    def call(self,
             query: tf.Tensor,
             key_value: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             training: bool = False,
             return_attention_weights: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Apply cross-attention.
        
        Args:
            query: Query tensor [batch_size, target_seq_len, d_model]
            key_value: Key and value tensor [batch_size, source_seq_len, d_model]
            mask: Attention mask [batch_size, 1, 1, source_seq_len]
            training: Training mode
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        return self.attention(
            query=query,
            key=key_value,
            value=key_value,
            mask=mask,
            training=training,
            return_attention_weights=return_attention_weights
        )
    
    def get_config(self):
        """Get layer configuration."""
        return self.attention.get_config()


class SelfAttention(tf.keras.layers.Layer):
    """
    Self-attention layer with peptide-specific optimizations.
    
    This layer is specifically designed for peptide sequence self-attention
    with optimizations for the typical sequence lengths (3-36 amino acids).
    """
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.1,
                 use_relative_position: bool = True,
                 max_relative_distance: int = 32,
                 name: str = "self_attention"):
        """
        Initialize self-attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for output
            attention_dropout: Dropout rate for attention weights
            use_relative_position: Whether to use relative positional encoding
            max_relative_distance: Maximum relative distance for relative encoding
            name: Layer name
        """
        super().__init__(name=name)
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            use_relative_position=use_relative_position,
            max_relative_distance=max_relative_distance,
            causal=False,
            name=f"{name}_mha"
        )
    
    def call(self,
             inputs: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             training: bool = False,
             return_attention_weights: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Apply self-attention.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, 1, 1, seq_len]
            training: Training mode
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        return self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            mask=mask,
            training=training,
            return_attention_weights=return_attention_weights
        )
    
    def get_config(self):
        """Get layer configuration."""
        return self.attention.get_config()


class CausalSelfAttention(tf.keras.layers.Layer):
    """
    Causal self-attention layer for autoregressive generation.
    
    This layer is used in the decoder part of TWAE-MMD for autoregressive
    peptide sequence generation.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.1,
                 name: str = "causal_self_attention"):
        """
        Initialize causal self-attention layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for output
            attention_dropout: Dropout rate for attention weights
            name: Layer name
        """
        super().__init__(name=name)
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            causal=True,
            name=f"{name}_mha"
        )
    
    def call(self,
             inputs: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             training: bool = False,
             return_attention_weights: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Apply causal self-attention.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, 1, 1, seq_len]
            training: Training mode
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        return self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            mask=mask,
            training=training,
            return_attention_weights=return_attention_weights
        )
    
    def get_config(self):
        """Get layer configuration."""
        return self.attention.get_config()

def create_attention_mask(input_ids: tf.Tensor, 
                         pad_token_id: int = 0,
                         dtype: tf.DType = None) -> tf.Tensor:
    """
    Create attention mask from input IDs with mixed precision support.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        pad_token_id: Padding token ID
        dtype: Target dtype for the mask (if None, uses input_ids dtype)
        
    Returns:
        Attention mask [batch_size, 1, 1, seq_len]
    """
    # Create mask (1 for valid tokens, 0 for padding)
    # FIXED: Use appropriate dtype for mixed precision compatibility
    if dtype is None:
        policy = tf.keras.mixed_precision.global_policy()
        dtype = policy.compute_dtype
    mask = tf.cast(tf.not_equal(input_ids, pad_token_id), dtype)
    
    # Reshape for attention: [batch_size, 1, 1, seq_len]
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_causal_mask(seq_len: int) -> tf.Tensor:
    """
    Create causal (lower triangular) mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Causal mask [seq_len, seq_len]
    """
    return tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


def create_attention_layer(config, 
                          attention_type: str = "self",
                          name: str = "attention") -> tf.keras.layers.Layer:
    """
    Create attention layer from configuration.
    
    Args:
        config: TWAE-MMD configuration object
        attention_type: Type of attention ('self', 'cross', 'causal')
        name: Layer name
        
    Returns:
        Attention layer
    """
    if attention_type == "self":
        return SelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            attention_dropout=config.attention_dropout,
            name=name
        )
    elif attention_type == "cross":
        return CrossAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            attention_dropout=config.attention_dropout,
            name=name
        )
    elif attention_type == "causal":
        return CausalSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            attention_dropout=config.attention_dropout,
            name=name
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


