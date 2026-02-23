"""
Production Transformer Layers for TWAE-MMD
Optimized for antimicrobial peptide sequence modeling

This module implements transformer blocks and custom layers specifically
designed for the TWAE-MMD architecture with peptide-specific optimizations.

"""

import tensorflow as tf
import numpy as np
import math
from typing import Optional, Tuple, Union, Callable

from .attention import SelfAttention, CrossAttention, CausalSelfAttention


class FeedForwardNetwork(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network with peptide-specific optimizations.
    
    This implementation includes:
    - GELU activation for better gradient flow
    - Dropout for regularization
    - Optional layer scaling for training stability
    """
    
    def __init__(self,
                 d_model: int = 256,
                 d_ff: int = 1024,
                 dropout_rate: float = 0.1,
                 activation: str = "gelu",
                 use_bias: bool = True,
                 use_layer_scale: bool = False,
                 layer_scale_init: float = 1e-4,
                 name: str = "feed_forward"):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout_rate: Dropout rate
            activation: Activation function ('gelu', 'relu', 'swish')
            use_bias: Whether to use bias in linear layers
            use_layer_scale: Whether to use layer scaling
            layer_scale_init: Initial value for layer scale
            name: Layer name
        """
        super().__init__(name=name)
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        
        # First linear layer
        self.dense1 = tf.keras.layers.Dense(
            d_ff, 
            use_bias=use_bias,
            name=f"{name}_dense1"
        )
        
        # Activation function
        if activation == "gelu":
            self.activation_fn = tf.keras.activations.gelu
        elif activation == "relu":
            self.activation_fn = tf.keras.activations.relu
        elif activation == "swish":
            self.activation_fn = tf.keras.activations.swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Second linear layer
        self.dense2 = tf.keras.layers.Dense(
            d_model,
            use_bias=use_bias,
            name=f"{name}_dense2"
        )
        
        # Layer scaling (if enabled)
        if use_layer_scale:
            self.layer_scale = self.add_weight(
                name="layer_scale",
                shape=(d_model,),
                initializer=tf.keras.initializers.Constant(layer_scale_init),
                trainable=True
            )
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Apply feed-forward network.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, d_model]
            training: Training mode
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # First linear transformation
        x = self.dense1(inputs)
        
        # Activation
        x = self.activation_fn(x)
        
        # Dropout
        x = self.dropout(x, training=training)
        
        # Second linear transformation
        x = self.dense2(x)
        
        # Layer scaling (if enabled)
        if self.use_layer_scale:
            x = x * self.layer_scale
        
        return x
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'use_layer_scale': self.use_layer_scale,
            'layer_scale_init': self.layer_scale_init,
        })
        return config


class StochasticDepth(tf.keras.layers.Layer):
    """
    Stochastic depth layer for regularization.
    
    Randomly drops entire layers during training to improve generalization
    and reduce overfitting in deep transformer models.
    """
    
    def __init__(self, 
                 drop_rate: float = 0.1,
                 name: str = "stochastic_depth"):
        """
        Initialize stochastic depth layer.
        
        Args:
            drop_rate: Probability of dropping the layer
            name: Layer name
        """
        super().__init__(name=name)
        self.drop_rate = drop_rate
    
    def call(self, 
             inputs: tf.Tensor, 
             residual: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        """
        Apply stochastic depth.
        
        Args:
            inputs: Input tensor from the layer
            residual: Residual connection tensor
            training: Training mode
            
        Returns:
            Output tensor with stochastic depth applied
        """
        if not training or self.drop_rate == 0.0:
            return inputs + residual
        
        # Random drop decision
        batch_size = tf.shape(inputs)[0]
        random_tensor = tf.random.uniform([batch_size, 1, 1])
        keep_prob = 1.0 - self.drop_rate
        
        # Binary mask - Cast to same dtype as inputs for mixed precision compatibility
        binary_mask = tf.cast(random_tensor < keep_prob, inputs.dtype)
        
        # Apply stochastic depth - Cast keep_prob to same dtype as inputs
        keep_prob_cast = tf.cast(keep_prob, inputs.dtype)
        output = inputs * binary_mask / keep_prob_cast + residual
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({'drop_rate': self.drop_rate})
        return config


class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block optimized for peptide sequences.
    
    This block includes:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Layer normalization
    - Optional stochastic depth
    - Optional layer scaling
    """
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.1,
                 stochastic_depth_rate: float = 0.0,
                 use_layer_scale: bool = False,
                 layer_scale_init: float = 1e-4,
                 norm_first: bool = True,
                 activation: str = "gelu",
                 name: str = "transformer_block"):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout_rate: Dropout rate
            attention_dropout: Attention dropout rate
            stochastic_depth_rate: Stochastic depth drop rate
            use_layer_scale: Whether to use layer scaling
            layer_scale_init: Initial value for layer scale
            norm_first: Whether to apply layer norm before attention/FFN
            activation: Activation function for FFN
            name: Layer name
        """
        super().__init__(name=name)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.norm_first = norm_first
        self.activation = activation
        
        # Multi-head self-attention
        self.self_attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            name=f"{name}_self_attention"
        )
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation=activation,
            use_layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init,
            name=f"{name}_ffn"
        )
        
        # Layer normalization
        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_norm1"
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_norm2"
        )
        
        # Stochastic depth (if enabled)
        if stochastic_depth_rate > 0.0:
            self.stochastic_depth = StochasticDepth(
                drop_rate=stochastic_depth_rate,
                name=f"{name}_stochastic_depth"
            )
        else:
            self.stochastic_depth = None
    
    def call(self,
             inputs: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             training: bool = False,
             return_attention_weights: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Apply transformer block.
        
        Args:
            inputs: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, 1, 1, seq_len]
            training: Training mode
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        # Self-attention with residual connection
        if self.norm_first:
            # Pre-norm: LayerNorm -> Attention -> Residual
            attn_input = self.norm1(inputs)
            if return_attention_weights:
                attn_output, attention_weights = self.self_attention(
                    attn_input, mask=mask, training=training, return_attention_weights=True
                )
            else:
                attn_output = self.self_attention(
                    attn_input, mask=mask, training=training
                )
            
            # Apply stochastic depth or regular residual
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(attn_output, inputs, training=training)
            else:
                x = attn_output + inputs
        else:
            # Post-norm: Attention -> Residual -> LayerNorm
            if return_attention_weights:
                attn_output, attention_weights = self.self_attention(
                    inputs, mask=mask, training=training, return_attention_weights=True
                )
            else:
                attn_output = self.self_attention(
                    inputs, mask=mask, training=training
                )
            
            # Apply stochastic depth or regular residual
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(attn_output, inputs, training=training)
            else:
                x = attn_output + inputs
            
            x = self.norm1(x)
        
        # Feed-forward with residual connection
        if self.norm_first:
            # Pre-norm: LayerNorm -> FFN -> Residual
            ffn_input = self.norm2(x)
            ffn_output = self.feed_forward(ffn_input, training=training)
            
            # Apply stochastic depth or regular residual
            if self.stochastic_depth is not None:
                output = self.stochastic_depth(ffn_output, x, training=training)
            else:
                output = ffn_output + x
        else:
            # Post-norm: FFN -> Residual -> LayerNorm
            ffn_output = self.feed_forward(x, training=training)
            
            # Apply stochastic depth or regular residual
            if self.stochastic_depth is not None:
                output = self.stochastic_depth(ffn_output, x, training=training)
            else:
                output = ffn_output + x
            
            output = self.norm2(output)
        
        if return_attention_weights:
            return output, attention_weights
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'use_layer_scale': self.use_layer_scale,
            'layer_scale_init': self.layer_scale_init,
            'norm_first': self.norm_first,
            'activation': self.activation,
        })
        return config


class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer decoder block for autoregressive generation.
    
    This block includes:
    - Causal self-attention
    - Cross-attention (encoder-decoder)
    - Feed-forward network
    - Residual connections and layer normalization
    """
    
    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 dropout_rate: float = 0.1,
                 attention_dropout: float = 0.1,
                 stochastic_depth_rate: float = 0.0,
                 use_layer_scale: bool = False,
                 layer_scale_init: float = 1e-4,
                 norm_first: bool = True,
                 activation: str = "gelu",
                 name: str = "decoder_block"):
        """
        Initialize decoder block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout_rate: Dropout rate
            attention_dropout: Attention dropout rate
            stochastic_depth_rate: Stochastic depth drop rate
            use_layer_scale: Whether to use layer scaling
            layer_scale_init: Initial value for layer scale
            norm_first: Whether to apply layer norm before attention/FFN
            activation: Activation function for FFN
            name: Layer name
        """
        super().__init__(name=name)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.norm_first = norm_first
        self.activation = activation
        
        # Causal self-attention
        self.causal_self_attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            name=f"{name}_causal_self_attention"
        )
        
        # Cross-attention (encoder-decoder)
        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            name=f"{name}_cross_attention"
        )
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation=activation,
            use_layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init,
            name=f"{name}_ffn"
        )
        
        # Layer normalization
        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_norm1"
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_norm2"
        )
        self.norm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_norm3"
        )
        
        # Stochastic depth (if enabled)
        if stochastic_depth_rate > 0.0:
            self.stochastic_depth = StochasticDepth(
                drop_rate=stochastic_depth_rate,
                name=f"{name}_stochastic_depth"
            )
        else:
            self.stochastic_depth = None
    
    def call(self,
             inputs: tf.Tensor,
             encoder_outputs: tf.Tensor,
             self_attention_mask: Optional[tf.Tensor] = None,
             cross_attention_mask: Optional[tf.Tensor] = None,
             training: bool = False) -> tf.Tensor:
        """
        Apply decoder block.
        
        Args:
            inputs: Input tensor [batch_size, target_seq_len, d_model]
            encoder_outputs: Encoder outputs [batch_size, source_seq_len, d_model]
            self_attention_mask: Self-attention mask
            cross_attention_mask: Cross-attention mask
            training: Training mode
            
        Returns:
            Output tensor [batch_size, target_seq_len, d_model]
        """
        # Causal self-attention
        if self.norm_first:
            attn_input = self.norm1(inputs)
            attn_output = self.causal_self_attention(
                attn_input, mask=self_attention_mask, training=training
            )
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(attn_output, inputs, training=training)
            else:
                x = attn_output + inputs
        else:
            attn_output = self.causal_self_attention(
                inputs, mask=self_attention_mask, training=training
            )
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(attn_output, inputs, training=training)
            else:
                x = attn_output + inputs
            x = self.norm1(x)
        
        # Cross-attention
        if self.norm_first:
            cross_input = self.norm2(x)
            cross_output = self.cross_attention(
                query=cross_input,
                key_value=encoder_outputs,
                mask=cross_attention_mask,
                training=training
            )
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(cross_output, x, training=training)
            else:
                x = cross_output + x
        else:
            cross_output = self.cross_attention(
                query=x,
                key_value=encoder_outputs,
                mask=cross_attention_mask,
                training=training
            )
            if self.stochastic_depth is not None:
                x = self.stochastic_depth(cross_output, x, training=training)
            else:
                x = cross_output + x
            x = self.norm2(x)
        
        # Feed-forward
        if self.norm_first:
            ffn_input = self.norm3(x)
            ffn_output = self.feed_forward(ffn_input, training=training)
            if self.stochastic_depth is not None:
                output = self.stochastic_depth(ffn_output, x, training=training)
            else:
                output = ffn_output + x
        else:
            ffn_output = self.feed_forward(x, training=training)
            if self.stochastic_depth is not None:
                output = self.stochastic_depth(ffn_output, x, training=training)
            else:
                output = ffn_output + x
            output = self.norm3(output)
        
        return output
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'use_layer_scale': self.use_layer_scale,
            'layer_scale_init': self.layer_scale_init,
            'norm_first': self.norm_first,
            'activation': self.activation,
        })
        return config


def create_transformer_block(d_model: int = 256,
                           num_heads: int = 8,
                           d_ff: int = 1024,
                           dropout_rate: float = 0.1,
                           attention_dropout: float = 0.1,
                           stochastic_depth_rate: float = 0.0,
                           use_layer_scale: bool = False,
                           layer_scale_init: float = 1e-6,
                           block_type: str = "encoder",
                           layer_idx: int = 0,
                           name: str = "transformer_block") -> tf.keras.layers.Layer:
    """
    Create transformer block.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout_rate: Dropout rate
        attention_dropout: Attention dropout rate
        stochastic_depth_rate: Stochastic depth rate
        use_layer_scale: Whether to use layer scaling
        layer_scale_init: Layer scale initialization value
        block_type: Type of block ('encoder', 'decoder')
        layer_idx: Layer index for stochastic depth scheduling
        name: Layer name
        
    Returns:
        Transformer block layer
    """
    if block_type == "encoder":
        return TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            stochastic_depth_rate=stochastic_depth_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init,
            name=name
        )
    elif block_type == "decoder":
        return DecoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            stochastic_depth_rate=stochastic_depth_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init=layer_scale_init,
            name=name
        )
    else:
        raise ValueError(f"Unknown block type: {block_type}")

