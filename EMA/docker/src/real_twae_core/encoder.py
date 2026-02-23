"""
Transformer Encoder for TWAE-MMD
Optimized for antimicrobial peptide sequence encoding

This module implements a transformer encoder specifically designed
for peptide sequences in the TWAE-MMD architecture, targeting >96% accuracy.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict, List

from .positional_encoding import PositionalEncoding
from .layers import TransformerBlock


class TransformerEncoder(tf.keras.layers.Layer):
    """
    Transformer encoder optimized for peptide sequences.
    
    This encoder processes peptide sequences (3-36 amino acids) and produces
    rich contextual representations for the TWAE-MMD latent space.
    
    Architecture:
    - Token embedding + positional encoding
    - Multiple transformer blocks with self-attention
    - Layer normalization and dropout
    - Global pooling for sequence-level representation
    """
    
    def __init__(self,
                 vocab_size: int = 25,
                 d_model: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 d_ff: int = 1024,
                 max_length: int = 37,
                 dropout_rate: float = 0.25,
                 attention_dropout: float = 0.15,
                 use_stochastic_depth: bool = True,
                 stochastic_depth_rate: float = 0.1,
                 use_layer_scale: bool = True,
                 layer_scale_init: float = 1e-4,
                 pooling_type: str = "attention",
                 name: str = "transformer_encoder"):
        """
        Initialize transformer encoder.
        
        Args:
            vocab_size: Vocabulary size (20 amino acids + 5 special tokens)
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            max_length: Maximum sequence length
            dropout_rate: Dropout rate
            attention_dropout: Attention dropout rate
            use_stochastic_depth: Whether to use stochastic depth
            stochastic_depth_rate: Stochastic depth rate
            use_layer_scale: Whether to use layer scaling
            layer_scale_init: Initial layer scale value
            pooling_type: Pooling type ('mean', 'max', 'attention', 'cls')
            name: Layer name
        """
        super().__init__(name=name)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.pooling_type = pooling_type
        
        # Token embedding layer
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            mask_zero=True,
            embeddings_initializer='uniform',
            name=f"{name}_token_embedding"
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            max_length=max_length,
            d_model=d_model,
            encoding_type="learned",  # Learned works better for peptides
            dropout_rate=dropout_rate,
            name=f"{name}_pos_encoding"
        )
        
        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            # Calculate stochastic depth rate for this layer
            if use_stochastic_depth:
                layer_drop_rate = stochastic_depth_rate * i / (num_layers - 1)
            else:
                layer_drop_rate = 0.0
            
            block = TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                attention_dropout=attention_dropout,
                stochastic_depth_rate=layer_drop_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init=layer_scale_init,
                norm_first=True,  # Pre-norm for better training stability
                name=f"{name}_block_{i}"
            )
            self.transformer_blocks.append(block)
        
        # Final layer normalization
        self.final_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_final_norm"
        )
        
        # Pooling layer
        if pooling_type == "attention":
            self.pooling_layer = AttentionPooling(
                d_model=d_model,
                name=f"{name}_attention_pooling"
            )
        elif pooling_type == "cls":
            # Add CLS token embedding
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, d_model),
                initializer="random_normal",
                trainable=True
            )
        
        # Dropout for final output
        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def _create_padding_mask(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Padding mask [batch_size, 1, 1, seq_len]
        """
        # Create mask (1 for valid tokens, 0 for padding)
        # FIXED: Cast to model's compute dtype for mixed precision compatibility
        mask = tf.cast(tf.not_equal(input_ids, 0), self.compute_dtype)
        
        # Reshape for attention: [batch_size, 1, 1, seq_len]
        return mask[:, tf.newaxis, tf.newaxis, :]
    
    def _apply_pooling(self, 
                      sequence_output: tf.Tensor,
                      attention_mask: tf.Tensor) -> tf.Tensor:
        """
        Apply pooling to get sequence-level representation.
        
        Args:
            sequence_output: Sequence outputs [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Pooled output [batch_size, d_model]
        """
        if self.pooling_type == "mean":
            # Mean pooling with masking
            mask = tf.squeeze(attention_mask, axis=[1, 2])  # [batch_size, seq_len]
            mask_expanded = mask[:, :, tf.newaxis]  # [batch_size, seq_len, 1]
            
            masked_output = sequence_output * mask_expanded
            sum_output = tf.reduce_sum(masked_output, axis=1)
            sum_mask = tf.reduce_sum(mask_expanded, axis=1)
            
            return sum_output / tf.maximum(sum_mask, 1e-9)
        
        elif self.pooling_type == "max":
            # Max pooling with masking
            mask = tf.squeeze(attention_mask, axis=[1, 2])  # [batch_size, seq_len]
            mask_expanded = mask[:, :, tf.newaxis]  # [batch_size, seq_len, 1]
            
            # Set padded positions to very negative values
            masked_output = sequence_output + (1.0 - mask_expanded) * -1e9
            
            return tf.reduce_max(masked_output, axis=1)
        
        elif self.pooling_type == "attention":
            # Attention-based pooling
            return self.pooling_layer(sequence_output, attention_mask)
        
        elif self.pooling_type == "cls":
            # Use CLS token representation (first token)
            return sequence_output[:, 0, :]
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def call(self,
             input_ids: tf.Tensor,
             attention_mask: Optional[tf.Tensor] = None,
             training: bool = False,
             return_all_layers: bool = False) -> tf.Tensor:
        """
        Forward pass of transformer encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            training: Training mode
            return_all_layers: Whether to return outputs from all layers
            
        Returns:
            Encoded representation [batch_size, d_model] or list of layer outputs
        """
        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self._create_padding_mask(input_ids)
        else:
            # Ensure attention_mask has proper shape and type
            # FIXED: Cast to model's compute dtype for mixed precision compatibility
            attention_mask = tf.cast(attention_mask, self.compute_dtype)
            
            # Check if attention_mask needs reshaping
            mask_shape = tf.shape(attention_mask)
            if len(attention_mask.shape) == 2:
                # Reshape from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            elif len(attention_mask.shape) == 1:
                # Handle edge case where mask might be 1D
                attention_mask = tf.expand_dims(attention_mask, 0)
                attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            # If already 4D, keep as is
        
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Add CLS token if using CLS pooling
        if self.pooling_type == "cls":
            cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
            x = tf.concat([cls_tokens, x], axis=1)
            
            # Update attention mask for CLS token
            cls_mask = tf.ones((batch_size, 1, 1, 1), dtype=tf.float32)
            attention_mask = tf.concat([cls_mask, attention_mask], axis=-1)
        
        # Positional encoding
        x = self.positional_encoding(x, training=training)
        
        # Apply transformer blocks
        layer_outputs = []
        for block in self.transformer_blocks:
            x = block(x, mask=attention_mask, training=training)
            if return_all_layers:
                layer_outputs.append(x)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        if return_all_layers:
            layer_outputs.append(x)
            return layer_outputs
        
        # Apply pooling to get sequence-level representation
        pooled_output = self._apply_pooling(x, attention_mask)
        
        # Final dropout
        pooled_output = self.output_dropout(pooled_output, training=training)
        
        return pooled_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'max_length': self.max_length,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'use_layer_scale': self.use_layer_scale,
            'layer_scale_init': self.layer_scale_init,
            'pooling_type': self.pooling_type,
        })
        return config


class AttentionPooling(tf.keras.layers.Layer):
    """
    Attention-based pooling for sequence-level representation.
    
    This layer learns to attend to the most important tokens in the sequence
    for creating a fixed-size representation.
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 name: str = "attention_pooling"):
        """
        Initialize attention pooling layer.
        
        Args:
            d_model: Model dimension
            name: Layer name
        """
        super().__init__(name=name)
        
        self.d_model = d_model
        
        # Attention projection
        self.attention_projection = tf.keras.layers.Dense(
            1,
            use_bias=False,
            name=f"{name}_attention_projection"
        )
    
    def call(self, 
             sequence_output: tf.Tensor,
             attention_mask: tf.Tensor) -> tf.Tensor:
        """
        Apply attention pooling.
        
        Args:
            sequence_output: Sequence outputs [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Pooled output [batch_size, d_model]
        """
        # Compute attention scores
        attention_scores = self.attention_projection(sequence_output)  # [batch_size, seq_len, 1]
        attention_scores = tf.squeeze(attention_scores, axis=-1)  # [batch_size, seq_len]
        
        # Apply mask to attention scores
        mask = tf.squeeze(attention_mask, axis=[1, 2])  # [batch_size, seq_len]
        # FIXED: Cast ALL tensors to same dtype as attention_scores for mixed precision compatibility
        mask_dtype = attention_scores.dtype
        mask_cast = tf.cast(mask, mask_dtype)
        one_val = tf.cast(1.0, mask_dtype)
        neg_val = tf.cast(-1e9, mask_dtype)
        attention_scores = attention_scores + (one_val - mask_cast) * neg_val
        
        # Compute attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # [batch_size, seq_len]
        attention_weights = tf.expand_dims(attention_weights, axis=-1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        pooled_output = tf.reduce_sum(sequence_output * attention_weights, axis=1)
        
        return pooled_output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config


def create_encoder(config, name: str = "encoder") -> TransformerEncoder:
    """
    Create transformer encoder from configuration.
    
    Args:
        config: TWAE-MMD configuration object
        name: Encoder name
        
    Returns:
        TransformerEncoder instance
    """
    return TransformerEncoder(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_length=config.max_length,
        dropout_rate=config.dropout_rate,
        attention_dropout=config.attention_dropout,
        use_stochastic_depth=config.use_stochastic_depth,
        stochastic_depth_rate=config.stochastic_depth_rate,
        use_layer_scale=config.use_layer_scale,
        layer_scale_init=config.layer_scale_init,
        pooling_type="attention",  # Attention pooling works best for peptides
        name=name
    )


