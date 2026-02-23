"""
Transformer Decoder for TWAE-MMD
Optimized for antimicrobial peptide sequence generation

This module implements a GPT-like transformer decoder specifically designed
for autoregressive peptide generation in the TWAE-MMD architecture.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Dict, List

from .positional_encoding import PositionalEncoding
from .layers import DecoderBlock


class TransformerDecoder(tf.keras.layers.Layer):
    """
    Transformer decoder optimized for peptide sequence generation.
    
    This decoder generates peptide sequences autoregressively from latent
    representations, with optional encoder cross-attention for reconstruction.
    
    Architecture:
    - Token embedding + positional encoding
    - Multiple decoder blocks with causal self-attention
    - Optional cross-attention to encoder outputs
    - Output projection to vocabulary
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
                 use_cross_attention: bool = True,
                 name: str = "transformer_decoder"):
        """
        Initialize transformer decoder.
        
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
            use_cross_attention: Whether to use cross-attention to encoder
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
        self.use_cross_attention = use_cross_attention
        
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
        
        # Latent projection (from latent space to decoder input)
        self.latent_projection = tf.keras.layers.Dense(
            d_model,
            activation='tanh',
            name=f"{name}_latent_projection"
        )
        
        # Decoder blocks
        self.decoder_blocks = []
        for i in range(num_layers):
            # Calculate stochastic depth rate for this layer
            if use_stochastic_depth:
                layer_drop_rate = stochastic_depth_rate * i / (num_layers - 1)
            else:
                layer_drop_rate = 0.0
            
            block = DecoderBlock(
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
            self.decoder_blocks.append(block)
        
        # Final layer normalization
        self.final_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{name}_final_norm"
        )
        
        # Output projection to vocabulary
        self.output_projection = tf.keras.layers.Dense(
            vocab_size,
            use_bias=False,  # No bias for output projection
            name=f"{name}_output_projection"
        )
        
        # Dropout for final output
        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def _create_causal_mask(self, seq_len: int) -> tf.Tensor:
        """
        Create causal (lower triangular) mask for autoregressive attention.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len]
        """
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]
    
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
    
    def _combine_masks(self, 
                      padding_mask: tf.Tensor,
                      causal_mask: tf.Tensor) -> tf.Tensor:
        """
        Combine padding and causal masks.
        
        Args:
            padding_mask: Padding mask [batch_size, 1, 1, seq_len]
            causal_mask: Causal mask [1, 1, seq_len, seq_len]
            
        Returns:
            Combined mask [batch_size, 1, seq_len, seq_len]
        """
        # Expand padding mask to match causal mask dimensions
        batch_size = tf.shape(padding_mask)[0]
        seq_len = tf.shape(padding_mask)[-1]
        
        # Expand padding mask: [batch_size, 1, seq_len, seq_len]
        padding_mask_expanded = tf.tile(padding_mask, [1, 1, seq_len, 1])
        
        # Expand causal mask: [batch_size, 1, seq_len, seq_len]
        causal_mask_expanded = tf.tile(causal_mask, [batch_size, 1, 1, 1])
        
        # Combine masks (both must be 1 for valid attention)
        combined_mask = padding_mask_expanded * causal_mask_expanded
        
        return combined_mask
    
    def _prepare_latent_input(self, 
                             latent_vector: tf.Tensor,
                             target_seq_len: int) -> tf.Tensor:
        """
        Prepare latent vector for cross-attention.
        
        Args:
            latent_vector: Latent vector [batch_size, latent_dim]
            target_seq_len: Target sequence length
            
        Returns:
            Prepared latent input [batch_size, 1, d_model]
        """
        # Project latent vector to model dimension
        latent_projected = self.latent_projection(latent_vector)  # [batch_size, d_model]
        
        # Add sequence dimension
        latent_input = tf.expand_dims(latent_projected, axis=1)  # [batch_size, 1, d_model]
        
        return latent_input
    
    def call(self,
             input_ids: tf.Tensor,
             latent_vector: Optional[tf.Tensor] = None,
             encoder_outputs: Optional[tf.Tensor] = None,
             attention_mask: Optional[tf.Tensor] = None,
             encoder_attention_mask: Optional[tf.Tensor] = None,
             training: bool = False,
             use_cache: bool = False,
             past_key_values: Optional[List] = None) -> tf.Tensor:
        """
        Forward pass of transformer decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            latent_vector: Latent vector [batch_size, latent_dim] (optional)
            encoder_outputs: Encoder outputs [batch_size, encoder_seq_len, d_model] (optional)
            attention_mask: Attention mask [batch_size, seq_len] (optional)
            encoder_attention_mask: Encoder attention mask [batch_size, encoder_seq_len] (optional)
            training: Training mode
            use_cache: Whether to use caching for generation (not implemented)
            past_key_values: Past key-value pairs for caching (not implemented)
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]
        
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Positional encoding
        x = self.positional_encoding(x, training=training)
        
        # Create attention masks
        if attention_mask is None:
            padding_mask = self._create_padding_mask(input_ids)
        else:
            padding_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        
        causal_mask = self._create_causal_mask(seq_len)
        self_attention_mask = self._combine_masks(padding_mask, causal_mask)
        
        # Prepare cross-attention inputs
        cross_attention_inputs = None
        cross_attention_mask = None
        
        if self.use_cross_attention:
            if latent_vector is not None:
                # Use latent vector for cross-attention
                cross_attention_inputs = self._prepare_latent_input(latent_vector, seq_len)
                # Create mask for latent input (always attend to latent)
                cross_attention_mask = tf.ones((batch_size, 1, 1, 1), dtype=tf.float32)
            elif encoder_outputs is not None:
                # Use encoder outputs for cross-attention
                cross_attention_inputs = encoder_outputs
                if encoder_attention_mask is not None:
                    cross_attention_mask = encoder_attention_mask[:, tf.newaxis, tf.newaxis, :]
                else:
                    encoder_seq_len = tf.shape(encoder_outputs)[1]
                    cross_attention_mask = tf.ones((batch_size, 1, 1, encoder_seq_len), dtype=tf.float32)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            if self.use_cross_attention and cross_attention_inputs is not None:
                x = block(
                    inputs=x,
                    encoder_outputs=cross_attention_inputs,
                    self_attention_mask=self_attention_mask,
                    cross_attention_mask=cross_attention_mask,
                    training=training
                )
            else:
                # Self-attention only (for pure generation)
                x = block(
                    inputs=x,
                    encoder_outputs=x,  # Use self as encoder outputs
                    self_attention_mask=self_attention_mask,
                    cross_attention_mask=self_attention_mask,
                    training=training
                )
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Output dropout
        x = self.output_dropout(x, training=training)
        
        # Output projection to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self,
                 start_token_id: int,
                 latent_vector: Optional[tf.Tensor] = None,
                 max_length: int = 37,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 1.0,
                 pad_token_id: int = 0,
                 eos_token_id: int = 4) -> tf.Tensor:
        """
        Generate peptide sequences autoregressively.
        
        Args:
            start_token_id: Start token ID (usually CLS token)
            latent_vector: Latent vector [batch_size, latent_dim] (optional)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Top-p (nucleus) sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated sequences [batch_size, seq_len]
        """
        if latent_vector is not None:
            batch_size = tf.shape(latent_vector)[0]
        else:
            batch_size = 1
        
        # Initialize with start token
        generated = tf.fill([batch_size, 1], start_token_id)
        
        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.call(
                input_ids=generated,
                latent_vector=latent_vector,
                training=False
            )
            
            # Get logits for next token (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = tf.nn.top_k(next_token_logits, k=top_k)
                next_token_logits = tf.where(
                    next_token_logits < tf.expand_dims(top_k_logits[:, -1], axis=1),
                    tf.fill(tf.shape(next_token_logits), -float('inf')),
                    next_token_logits
                )
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits = tf.sort(next_token_logits, direction='DESCENDING')
                sorted_probs = tf.nn.softmax(sorted_logits)
                cumulative_probs = tf.cumsum(sorted_probs, axis=1)
                
                # Find cutoff index
                cutoff_mask = cumulative_probs <= top_p
                cutoff_indices = tf.reduce_sum(tf.cast(cutoff_mask, tf.int32), axis=1, keepdims=True)
                cutoff_indices = tf.maximum(cutoff_indices, 1)  # Keep at least one token
                
                # Create mask for valid tokens
                indices = tf.range(tf.shape(sorted_logits)[1])[tf.newaxis, :]
                valid_mask = indices < cutoff_indices
                
                # Apply mask to original logits
                sorted_indices = tf.argsort(next_token_logits, direction='DESCENDING')
                valid_mask_original = tf.gather(valid_mask, sorted_indices, batch_dims=1)
                
                next_token_logits = tf.where(
                    valid_mask_original,
                    next_token_logits,
                    tf.fill(tf.shape(next_token_logits), -float('inf'))
                )
            
            # Sample next token
            next_token_probs = tf.nn.softmax(next_token_logits)
            next_token = tf.random.categorical(tf.math.log(next_token_probs), num_samples=1)
            next_token = tf.cast(next_token, tf.int32)
            
            # Append to generated sequence
            generated = tf.concat([generated, next_token], axis=1)
            
            # Check for EOS token (early stopping)
            if tf.reduce_all(tf.equal(next_token[:, 0], eos_token_id)):
                break
        
        return generated
    
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
            'use_cross_attention': self.use_cross_attention,
        })
        return config


def create_decoder(config, 
                  use_cross_attention: bool = True,
                  name: str = "decoder") -> TransformerDecoder:
    """
    Create transformer decoder from configuration.
    
    Args:
        config: TWAE-MMD configuration object
        use_cross_attention: Whether to use cross-attention
        name: Decoder name
        
    Returns:
        TransformerDecoder instance
    """
    return TransformerDecoder(
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
        use_cross_attention=use_cross_attention,
        name=name
    )


