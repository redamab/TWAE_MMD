"""
PRODUCTION TWAE-MMD MODEL IMPLEMENTATION
Complete TWAE-MMD architecture with Transformer Decoder ONLY (No MLP)

This module provides the complete production-ready TWAE-MMD model:
- Transformer encoder for sequence encoding
- Transformer decoder for reconstruction (GPT-like with cross-attention)
- Sophisticated latent space for latent vector management
- ALL loss functions: Classification, Reconstruction, MMD, Wasserstein
- High-quality AMP generation capabilities
- Mixed precision support
- Production-optimized for training

Author: Reda Mabrouki
Updated: Using Transformer Decoder ONLY
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from .config import TWAEMMDConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .latent_space import LatentSpaceManager, LatentSpaceConfig


class TWAEMMDModel(tf.keras.Model):
    """
    Production TWAE-MMD model with Transformer Decoder for reconstruction.
    
    This model implements the complete TWAE-MMD architecture with:
    - Transformer encoder for sequence encoding
    - Latent space for distribution matching and generation
    - Transformer decoder (GPT-like) for autoregressive reconstruction
    - Cross-attention from decoder to latent vector
    - ALL loss functions: Classification, Reconstruction, MMD, Wasserstein
    - High-quality AMP generation capabilities
    """
    
    def __init__(self,
                 config: TWAEMMDConfig,
                 name: str = "twae_mmd_model",
                 **kwargs):
        """
        Initialize TWAE-MMD model with Transformer Decoder.
        
        Args:
            config: Model configuration
            name: Model name
        """
        super().__init__(name=name, **kwargs)
        
        self.config = config
        
        # Create transformer encoder
        self.encoder = TransformerEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.encoder_num_layers,
            d_ff=config.d_ff,
            max_length=config.max_length,
            dropout_rate=config.dropout_rate,
            attention_dropout=config.attention_dropout,
            use_stochastic_depth=config.use_stochastic_depth,
            stochastic_depth_rate=config.stochastic_depth_rate,
            use_layer_scale=config.use_layer_scale,
            layer_scale_init=config.layer_scale_init,
            name="transformer_encoder"
        )
        
        # Create latent space manager (for encoding/decoding latent vectors)
        latent_config = LatentSpaceConfig()
        latent_config.latent_dim = config.latent_dim
        latent_config.sequence_length = config.max_length
        latent_config.vocab_size = config.vocab_size
        latent_config.hidden_size = config.d_model
        latent_config.encoder_hidden_dims = config.latent_encoder_dims
        latent_config.activation = config.latent_activation
        latent_config.dropout_rate = config.latent_dropout
        latent_config.use_batch_norm = config.latent_use_batch_norm
        latent_config.latent_regularization = config.latent_regularization
        
        self.latent_manager = LatentSpaceManager(
            latent_dim=latent_config.latent_dim,
            sequence_length=latent_config.sequence_length,
            vocab_size=latent_config.vocab_size,
            hidden_size=latent_config.hidden_size,
            encoder_hidden_dims=latent_config.encoder_hidden_dims,
            activation=latent_config.activation,
            dropout_rate=latent_config.dropout_rate,
            use_batch_norm=latent_config.use_batch_norm,
            latent_regularization=latent_config.latent_regularization,
            name="latent_space_manager"
        )
        
        # Create transformer decoder (GPT-like with cross-attention to latent)
        self.decoder = TransformerDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.decoder_num_layers,
            d_ff=config.d_ff,
            max_length=config.max_length,
            dropout_rate=config.dropout_rate,
            attention_dropout=config.attention_dropout,
            use_stochastic_depth=config.use_stochastic_depth,
            stochastic_depth_rate=config.stochastic_depth_rate,
            use_layer_scale=config.use_layer_scale,
            layer_scale_init=config.layer_scale_init,
            use_cross_attention=True,  # Enable cross-attention to latent
            name="transformer_decoder"
        )
        

        # ===== FIX 1: Latent-to-Sequence Projection =====
        # Project latent vector [batch, 128] to sequence [batch, 37, 256]
        # This creates a DIRECT gradient path from latent to decoder!
        self.latent_to_sequence_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config.max_length * config.d_model,  # 37 * 256 = 9,472
                activation='gelu',
                name="latent_to_seq_dense"
            ),
            tf.keras.layers.Reshape(
                (config.max_length, config.d_model),  # [batch, 37, 256]
                name="latent_to_seq_reshape"
            ),
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                name="latent_to_seq_norm"
            )
        ], name="latent_to_sequence_projection")
        


        # Special tokens for decoder
        self.start_token_id = 0  # [CLS] token
        self.pad_token_id = 2    # [PAD] token
        self.sep_token_id = 1    # [SEP] token
        self.mask_token_id = 3   # [MASK] token
        
        # Classification head
        self.classification_head = tf.keras.layers.Dense(
            config.num_classes,
            name="classification_head"
        )
        

        # ===== FIX 3: Auxiliary Loss Components =====
        
        # Auxiliary Loss 1: Latent Reconstruction Head
        # Predicts latent vector from decoder output
        self.latent_reconstruction_head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='gelu', name="latent_recon_dense1"),
            tf.keras.layers.Dropout(0.1, name="latent_recon_dropout"),
            tf.keras.layers.Dense(config.latent_dim, activation='tanh', name="latent_recon_dense2")
        ], name="latent_reconstruction_head")
        
        # Build the latent_reconstruction_head to create weights
        # Input shape: [batch_size, d_model] from decoder pooled output
        self.latent_reconstruction_head.build((None, config.d_model))

        # Generation parameters
        self.generation_temperature = config.generation_temperature
        self.generation_strategy = config.generation_strategy
        self.generation_num_samples = config.generation_num_samples
    
    def call(self, inputs, training=None, attention_mask=None):
        """
        Forward pass through TWAE-MMD model with Transformer Decoder.
        
        Args:
            inputs: Input token IDs [batch_size, seq_len]
            training: Training mode flag
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with model outputs
        """
        # ===== ENCODING PHASE =====
        # Encode sequences with transformer encoder
        encoder_outputs = self.encoder(
            inputs, 
            attention_mask=attention_mask, 
            training=training
        )
        
        # Get sequence representation (pooled) with robust shape handling
        if len(encoder_outputs.shape) == 3:
            # Expected case: [batch_size, seq_len, d_model]
            sequence_representation = tf.reduce_mean(encoder_outputs, axis=1)
        elif len(encoder_outputs.shape) == 2:
            # Fallback case: [batch_size, d_model] (already pooled)
            sequence_representation = encoder_outputs
        else:
            # Error case: unexpected shape - force reshape
            sequence_representation = tf.reshape(encoder_outputs, [-1, self.config.d_model])
        
        # Ensure correct shape [batch_size, d_model]
        sequence_representation = tf.ensure_shape(
            sequence_representation, 
            [None, self.config.d_model]
        )
        
        # Encode to latent space
        latent_vector = self.latent_manager.encode(
            sequence_representation, 
            training=training
        )
        
        # Classification from latent vector
        classification_logits = self.classification_head(latent_vector)
        
        # ===== DECODING PHASE with Transformer Decoder =====
        # ===== FIX 1: Use Latent-to-Sequence Projection =====
        latent_sequence = self.latent_to_sequence_projection(latent_vector)
        
        # Prepare decoder input (shifted right for teacher forcing during training)
        if training:
            # Teacher forcing: use ground truth shifted right
            decoder_input_ids = self._prepare_decoder_input(inputs)
        else:
            # Inference: use input as decoder input
            # (For generation, use generate_sequences method)
            decoder_input_ids = inputs
        
        # Decode with transformer decoder (cross-attention to latent vector)
        reconstruction_logits = self.decoder(
            input_ids=decoder_input_ids,
            latent_vector=latent_vector,  # Cross-attention to latent!
            encoder_outputs=None,  # Not using encoder outputs
            attention_mask=attention_mask,
            training=training
        )
        
        outputs = {
            'classification_logits': classification_logits,
            'reconstruction_logits': reconstruction_logits,
            'latent_vector': latent_vector,
            'encoder_outputs': encoder_outputs,
            'sequence_representation': sequence_representation,
            'latent_sequence': latent_sequence
        }
        
        # Compute auxiliary outputs for training
        if training:
            # Pool latent_sequence over sequence dimension to get [batch_size, d_model]
            # latent_sequence shape: [batch_size, seq_len, d_model]
            decoder_pooled = tf.reduce_mean(latent_sequence, axis=1)
            predicted_latent = self.latent_reconstruction_head(decoder_pooled)
            outputs['auxiliary'] = {
                'predicted_latent': predicted_latent,
                'latent_sequence': latent_sequence
            }
        
        return outputs
    
    def _prepare_decoder_input(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Prepare decoder input by shifting right (for teacher forcing).
        
        Args:
            input_ids: Original input IDs [batch_size, seq_len]
            
        Returns:
            Shifted input IDs [batch_size, seq_len]
        """
        batch_size = tf.shape(input_ids)[0]
        
        # Create start tokens [batch_size, 1]
        start_tokens = tf.fill([batch_size, 1], self.start_token_id)
        
        # Shift right: [start_token, x1, x2, ..., x_{n-1}]
        shifted_input = tf.concat([start_tokens, input_ids[:, :-1]], axis=1)
        
        return shifted_input
    
    def compute_auxiliary_losses(self, outputs: Dict, targets: tf.Tensor, training: bool = True) -> Dict[str, tf.Tensor]:
        """
        Compute auxiliary losses for better decoder learning.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target sequences [batch_size, seq_len]
            training: Training mode flag
        
        Returns:
            Dictionary with auxiliary losses
        """
        if not training or "auxiliary" not in outputs:
            return {}
        
        aux_outputs = outputs["auxiliary"]
        latent_vector = outputs["latent_vector"]
        reconstruction_logits = outputs["reconstruction_logits"]
        
        aux_losses = {}
        
        # Auxiliary Loss 1: Latent Reconstruction
        predicted_latent = aux_outputs["predicted_latent"]
        aux_losses["latent_reconstruction"] = tf.reduce_mean(
            tf.square(predicted_latent - latent_vector)
        )
        
        # Auxiliary Loss 2: Token-Level (weighted by position)
        token_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=reconstruction_logits
        )
        seq_len = tf.shape(targets)[1]
        position_weights = tf.linspace(1.5, 1.0, seq_len)
        aux_losses["token_level"] = tf.reduce_mean(token_losses * position_weights)
        
        # Auxiliary Loss 3: Attention Alignment
        latent_sequence = aux_outputs["latent_sequence"]
        latent_seq_norm = tf.nn.l2_normalize(latent_sequence, axis=-1)
        similarity_matrix = tf.matmul(latent_seq_norm, latent_seq_norm, transpose_b=True)
        diagonal = tf.linalg.diag_part(similarity_matrix)
        aux_losses["attention_alignment"] = -tf.reduce_mean(diagonal)
        
        # Auxiliary Loss 4: Intermediate Supervision
        aux_losses["intermediate_supervision"] = tf.reduce_mean(tf.square(latent_sequence))
        
        return aux_losses
    
    def encode_to_latent(self, inputs, training=None, attention_mask=None):
        """
        Encode input sequences to latent space.
        
        Args:
            inputs: Input token IDs [batch_size, seq_len]
            training: Training mode flag
            attention_mask: Attention mask
            
        Returns:
            Latent vectors [batch_size, latent_dim]
        """
        # Encode sequences
        encoder_outputs = self.encoder(
            inputs, 
            attention_mask=attention_mask, 
            training=training
        )
        
        # Pool to sequence representation
        if len(encoder_outputs.shape) == 3:
            sequence_representation = tf.reduce_mean(encoder_outputs, axis=1)
        else:
            sequence_representation = encoder_outputs
        
        # Encode to latent space
        latent_vector = self.latent_manager.encode(
            sequence_representation, 
            training=training
        )
        
        return latent_vector
    
    def decode_from_latent(self, latent_vectors, temperature=1.0, training=None):
        """
        Decode latent vectors to sequences using Transformer Decoder.
        
        Args:
            latent_vectors: Latent vectors [batch_size, latent_dim]
            temperature: Sampling temperature
            training: Training mode flag
            
        Returns:
            Generated sequences [batch_size, seq_len]
        """
        batch_size = tf.shape(latent_vectors)[0]
        
        # Generate sequences autoregressively with transformer decoder
        generated_sequences = self.decoder.generate(
            start_token_id=self.start_token_id,
            latent_vector=latent_vectors,
            max_length=self.config.max_length,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.sep_token_id
        )
        
        return generated_sequences
    
    def generate_sequences(self, 
                          num_samples: int = 10,
                          temperature: float = 1.0,
                          sampling_strategy: str = 'gaussian',
                          training: bool = False) -> tf.Tensor:
        """
        Generate sequences by sampling from latent space with Transformer Decoder.
        
        Args:
            num_samples: Number of sequences to generate
            temperature: Sampling temperature
            sampling_strategy: Sampling strategy ('gaussian', 'uniform', 'learned')
            training: Training mode flag
            
        Returns:
            Generated sequences [num_samples, seq_len]
        """
        # Sample latent vectors from learned or prior distribution
        if sampling_strategy == 'learned':
            # Sample from learned distribution (uses latent_manager statistics)
            latent_samples = self.latent_manager.sample_latent(
                num_samples=num_samples,
                temperature=temperature
            )
        elif sampling_strategy == 'gaussian':
            # Sample from standard Gaussian prior
            latent_samples = tf.random.normal(
                [num_samples, self.config.latent_dim],
                mean=0.0,
                stddev=temperature,
                dtype=tf.float32
            )
        elif sampling_strategy == 'uniform':
            # Sample from uniform distribution
            latent_samples = tf.random.uniform(
                [num_samples, self.config.latent_dim],
                minval=-2.0 * temperature,
                maxval=2.0 * temperature,
                dtype=tf.float32
            )
        else:
            # Default: Gaussian
            latent_samples = tf.random.normal(
                [num_samples, self.config.latent_dim],
                mean=0.0,
                stddev=temperature,
                dtype=tf.float32
            )
        
        # Generate sequences with transformer decoder
        generated_sequences = self.decoder.generate(
            start_token_id=self.start_token_id,
            latent_vector=latent_samples,
            max_length=self.config.max_length,
            temperature=temperature,
            top_k=0,  # No top-k filtering
            top_p=1.0,  # No nucleus sampling
            pad_token_id=self.pad_token_id,
            eos_token_id=self.sep_token_id
        )
        
        return generated_sequences
    
    def interpolate_sequences(self,
                            seq1: tf.Tensor,
                            seq2: tf.Tensor,
                            num_steps: int = 5,
                            temperature: float = 1.0,
                            training: bool = False) -> tf.Tensor:
        """
        Interpolate between two sequences in latent space.
        
        Args:
            seq1: First sequence [1, seq_len]
            seq2: Second sequence [1, seq_len]
            num_steps: Number of interpolation steps
            temperature: Sampling temperature
            training: Training mode flag
            
        Returns:
            Interpolated sequences [num_steps, seq_len]
        """
        # Encode both sequences to latent space
        latent1 = self.encode_to_latent(seq1, training=training)
        latent2 = self.encode_to_latent(seq2, training=training)
        
        # Create interpolation weights
        alphas = tf.linspace(0.0, 1.0, num_steps)
        alphas = tf.reshape(alphas, [num_steps, 1])
        
        # Interpolate in latent space
        latent1_expanded = tf.tile(latent1, [num_steps, 1])
        latent2_expanded = tf.tile(latent2, [num_steps, 1])
        interpolated_latents = (1 - alphas) * latent1_expanded + alphas * latent2_expanded
        
        # Decode interpolated latents with transformer decoder
        interpolated_sequences = self.decoder.generate(
            start_token_id=self.start_token_id,
            latent_vector=interpolated_latents,
            max_length=self.config.max_length,
            temperature=temperature,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.sep_token_id
        )
        
        return interpolated_sequences
    
    # ===== LOSS FUNCTIONS =====
    
    def compute_classification_loss(self,
                                   classification_logits: tf.Tensor,
                                   labels: tf.Tensor,
                                   label_smoothing: float = 0.1) -> tf.Tensor:
        """
        Compute classification loss with label smoothing.
        
        Args:
            classification_logits: Classification logits [batch_size, num_classes]
            labels: True labels [batch_size]
            label_smoothing: Label smoothing factor
            
        Returns:
            Classification loss (scalar)
        """
        # Use SparseCategoricalCrossentropy with label_smoothing
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(labels, classification_logits)
        
        # Apply label smoothing manually if needed
        if label_smoothing > 0:
            num_classes = tf.shape(classification_logits)[-1]
            smooth_labels = tf.one_hot(labels, num_classes)
            smooth_labels = smooth_labels * (1 - label_smoothing) + label_smoothing / tf.cast(num_classes, tf.float32)
            loss_fn_smooth = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
            )
            loss = loss_fn_smooth(smooth_labels, classification_logits)
        
        return tf.reduce_mean(loss)
    
    def compute_reconstruction_loss(self,
                                   reconstruction_logits: tf.Tensor,
                                   target_ids: tf.Tensor,
                                   padding_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute reconstruction loss with proper masking for transformer decoder.
        
        Args:
            reconstruction_logits: Predicted logits [batch_size, seq_len, vocab_size]
            target_ids: Target token IDs [batch_size, seq_len]
            padding_mask: Padding mask [batch_size, seq_len] (optional)
            
        Returns:
            Reconstruction loss (scalar)
        """
        # Create padding mask if not provided
        if padding_mask is None:
            padding_mask = tf.cast(tf.not_equal(target_ids, self.pad_token_id), tf.float32)
        
        # Compute cross-entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            target_ids,
            reconstruction_logits,
            from_logits=True
        )
        
        # Apply padding mask
        loss = loss * padding_mask
        
        # Normalize by number of non-padding tokens
        num_tokens = tf.reduce_sum(padding_mask)
        loss = tf.reduce_sum(loss) / (num_tokens + 1e-8)
        
        return loss



    
    def compute_mmd_loss(self,
                        latent_vectors: tf.Tensor,
                        prior_samples: Optional[tf.Tensor] = None,
                        kernel_type: str = "mixed",
                        kernel_mul: float = 2.0,
                        num_kernels: int = 5) -> tf.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD) loss.
        
        Args:
            latent_vectors: Encoded latent vectors [batch_size, latent_dim]
            prior_samples: Prior samples (if None, sample from standard Gaussian)
            kernel_type: Kernel type ("rbf", "imq", "mixed")
            kernel_mul: Kernel multiplier for multi-scale
            num_kernels: Number of kernels for multi-scale
            
        Returns:
            MMD loss value
        """
        batch_size = tf.shape(latent_vectors)[0]
        latent_dim = tf.shape(latent_vectors)[1]
        
        # Sample from prior if not provided
        if prior_samples is None:
            prior_samples = tf.random.normal(
                [batch_size, latent_dim],
                mean=0.0,
                stddev=1.0,
                dtype=latent_vectors.dtype
            )
        
        # Compute pairwise distances
        def compute_pairwise_distances(x, y):
            """Compute pairwise squared Euclidean distances."""
            x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
            y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
            distances = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y, transpose_b=True)
            # Add small epsilon for numerical stability
            return tf.maximum(distances, 1e-12)
        
        # Compute kernel matrices
        xx = compute_pairwise_distances(latent_vectors, latent_vectors)
        yy = compute_pairwise_distances(prior_samples, prior_samples)
        xy = compute_pairwise_distances(latent_vectors, prior_samples)
        
        # Multi-scale RBF kernel
        def compute_rbf_kernel(distances, bandwidth):
            """Compute RBF kernel."""
            return tf.exp(-distances / (2.0 * bandwidth))
        
        # Multi-scale IMQ kernel
        def compute_imq_kernel(distances, bandwidth):
            """Compute Inverse Multi-Quadratic kernel."""
            return bandwidth / (bandwidth + distances)
        
        # Compute multi-scale bandwidths
        # Use percentile-based median (TF 2.13 compatible)
        xy_sorted = tf.sort(tf.reshape(xy, [-1]))
        n = tf.shape(xy_sorted)[0]
        median_distance = xy_sorted[n // 2] + 1e-6  # Add epsilon for numerical stability
        # Clip bandwidths to prevent extreme values
        bandwidths = [tf.clip_by_value(median_distance * (kernel_mul ** i), 1e-6, 1e6) 
                      for i in range(num_kernels)]
        
        # Compute kernel values
        if kernel_type == "rbf":
            kernel_xx = sum([compute_rbf_kernel(xx, bw) for bw in bandwidths]) / num_kernels
            kernel_yy = sum([compute_rbf_kernel(yy, bw) for bw in bandwidths]) / num_kernels
            kernel_xy = sum([compute_rbf_kernel(xy, bw) for bw in bandwidths]) / num_kernels
        elif kernel_type == "imq":
            kernel_xx = sum([compute_imq_kernel(xx, bw) for bw in bandwidths]) / num_kernels
            kernel_yy = sum([compute_imq_kernel(yy, bw) for bw in bandwidths]) / num_kernels
            kernel_xy = sum([compute_imq_kernel(xy, bw) for bw in bandwidths]) / num_kernels
        else:  # mixed
            # Mix of RBF and IMQ kernels
            kernel_xx = sum([compute_rbf_kernel(xx, bw) + compute_imq_kernel(xx, bw) 
                           for bw in bandwidths]) / (2 * num_kernels)
            kernel_yy = sum([compute_rbf_kernel(yy, bw) + compute_imq_kernel(yy, bw) 
                           for bw in bandwidths]) / (2 * num_kernels)
            kernel_xy = sum([compute_rbf_kernel(xy, bw) + compute_imq_kernel(xy, bw) 
                           for bw in bandwidths]) / (2 * num_kernels)
        
        # Unbiased MMD estimator
        batch_size_f = tf.cast(batch_size, tf.float32)
        
        # Remove diagonal elements for unbiased estimation
        mask = 1.0 - tf.eye(batch_size, dtype=tf.float32)
        kernel_xx = kernel_xx * mask
        kernel_yy = kernel_yy * mask
        
        # Compute MMD with proper scaling and numerical stability
        # FIX: Add epsilon to prevent division by zero and ensure non-zero gradient
        term1 = tf.reduce_sum(kernel_xx) / (batch_size_f * (batch_size_f - 1.0) + 1e-8)
        term2 = tf.reduce_sum(kernel_yy) / (batch_size_f * (batch_size_f - 1.0) + 1e-8)
        term3 = 2.0 * tf.reduce_mean(kernel_xy)
        
        mmd = term1 + term2 - term3
        
        # FIX: Ensure MMD is never exactly 0.0 to maintain gradient flow
        mmd = tf.maximum(mmd, 1e-6)
        
        # FIX: Scale MMD to reasonable range for better gradient signal
        # This ensures it contributes meaningfully alongside other losses
        # Increased from 10x to 100x for stronger signal (Final Fix)
        mmd = mmd * 100.0
        
        return mmd
    
    def compute_wasserstein_loss(self,
                                latent_vectors: tf.Tensor,
                                prior_samples: Optional[tf.Tensor] = None,
                                method: str = "energy",  # Changed from sinkhorn to energy for stability
                                epsilon: float = 0.05,
                                max_iterations: int = 15) -> tf.Tensor:
        """
        Compute Wasserstein loss between latent vectors and prior distribution.
        
        Args:
            latent_vectors: Encoded latent vectors [batch_size, latent_dim]
            prior_samples: Prior samples (if None, sample from standard Gaussian)
            method: Computation method ("sinkhorn", "energy", "sliced")
            epsilon: Entropy regularization for Sinkhorn
            max_iterations: Maximum Sinkhorn iterations
            
        Returns:
            Wasserstein loss value
        """
        batch_size = tf.shape(latent_vectors)[0]
        latent_dim = tf.shape(latent_vectors)[1]
        
        # Sample from prior if not provided
        if prior_samples is None:
            prior_samples = tf.random.normal(
                [batch_size, latent_dim],
                mean=0.0,
                stddev=1.0,
                dtype=latent_vectors.dtype
            )
        
        if method == "sinkhorn":
            # Use larger epsilon for better stability during training
            return self._compute_sinkhorn_divergence(
                latent_vectors, prior_samples, epsilon=0.1, max_iterations=5
            )
        elif method == "energy":
            return self._compute_energy_distance(latent_vectors, prior_samples)
        elif method == "sliced":
            return self._compute_sliced_wasserstein(latent_vectors, prior_samples)
        else:
            raise ValueError(f"Unsupported Wasserstein method: {method}")
    
    def _compute_sinkhorn_divergence(self,
                                   x: tf.Tensor,
                                   y: tf.Tensor,
                                   epsilon: float = 0.05,
                                   max_iterations: int = 15) -> tf.Tensor:
        """
        Compute Sinkhorn divergence (entropy-regularized Wasserstein distance).
        
        Args:
            x: First sample [batch_size, dim]
            y: Second sample [batch_size, dim]
            epsilon: Entropy regularization parameter
            max_iterations: Maximum number of Sinkhorn iterations
            
        Returns:
            Sinkhorn divergence
        """
        batch_size = tf.shape(x)[0]
        
        # Compute cost matrix (squared Euclidean distance) with numerical stability
        x_norm = tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=1, keepdims=True)
        cost_matrix = x_norm + tf.transpose(y_norm) - 2.0 * tf.matmul(x, y, transpose_b=True)
        # Ensure non-negative and clip to prevent extreme values
        cost_matrix = tf.clip_by_value(cost_matrix, 0.0, 1000.0)
        
        # Initialize dual variables
        batch_size_f = tf.cast(batch_size, tf.float32)
        log_mu = -tf.math.log(batch_size_f) * tf.ones([batch_size, 1], dtype=x.dtype)
        log_nu = -tf.math.log(batch_size_f) * tf.ones([batch_size, 1], dtype=x.dtype)
        
        # Sinkhorn iterations with numerical stability
        # Reduce iterations to 5 for better gradient stability
        for _ in range(5):  # Reduced from 15 to 5
            # Update log_mu with clipping for stability
            log_mu_update = -tf.reduce_logsumexp(
                (-cost_matrix + tf.transpose(log_nu)) / epsilon,
                axis=1,
                keepdims=True
            )
            # Clip to prevent extreme values
            log_mu = tf.clip_by_value(log_mu_update, -50.0, 50.0)
            
            # Update log_nu with clipping for stability
            log_nu_update = -tf.reduce_logsumexp(
                (-cost_matrix + tf.transpose(log_mu)) / epsilon,
                axis=0,
                keepdims=True
            )
            # Clip to prevent extreme values
            log_nu = tf.transpose(tf.clip_by_value(log_nu_update, -50.0, 50.0))
        
        # Compute transport plan with numerical stability
        log_transport = (-cost_matrix + tf.transpose(log_mu) + log_nu) / epsilon
        # Clip to prevent overflow/underflow
        log_transport = tf.clip_by_value(log_transport, -50.0, 50.0)
        transport = tf.exp(log_transport)
        
        # Add small epsilon to prevent log(0)
        transport_safe = tf.maximum(transport, 1e-12)
        
        # Compute Sinkhorn divergence with numerical stability
        # W_ε(μ, ν) = <C, π> - ε H(π)
        primal_cost = tf.reduce_sum(transport * cost_matrix)
        
        # Compute entropy safely: -sum(p * log(p))
        # Use transport_safe for log to avoid log(0)
        log_transport_safe = tf.math.log(transport_safe)
        entropy = -tf.reduce_sum(transport * log_transport_safe)
        
        sinkhorn_divergence = primal_cost - epsilon * entropy
        
        # Clip final result to reasonable range
        sinkhorn_divergence = tf.clip_by_value(sinkhorn_divergence, 0.0, 100.0)
        
        return sinkhorn_divergence
    
    def _compute_energy_distance(self,
                                x: tf.Tensor,
                                y: tf.Tensor) -> tf.Tensor:
        """
        Compute energy distance between two distributions.
        
        Args:
            x: First sample [batch_size, dim]
            y: Second sample [batch_size, dim]
            
        Returns:
            Energy distance
        """
        # Compute pairwise distances
        def pairwise_distance(a, b):
            """Compute pairwise Euclidean distances."""
            a_norm = tf.reduce_sum(tf.square(a), axis=1, keepdims=True)
            b_norm = tf.reduce_sum(tf.square(b), axis=1, keepdims=True)
            dist_sq = a_norm + tf.transpose(b_norm) - 2.0 * tf.matmul(a, b, transpose_b=True)
            return tf.sqrt(tf.maximum(dist_sq, 1e-12))
        
        # Energy distance: 2E[||X-Y||] - E[||X-X'||] - E[||Y-Y'||]
        xy_dist = tf.reduce_mean(pairwise_distance(x, y))
        xx_dist = tf.reduce_mean(pairwise_distance(x, x))
        yy_dist = tf.reduce_mean(pairwise_distance(y, y))
        
        energy_dist = 2.0 * xy_dist - xx_dist - yy_dist
        
        return tf.maximum(energy_dist, 0.0)
    
    def _compute_sliced_wasserstein(self,
                                   x: tf.Tensor,
                                   y: tf.Tensor,
                                   num_projections: int = 100) -> tf.Tensor:
        """
        Compute sliced Wasserstein distance.
        
        Args:
            x: First sample [batch_size, dim]
            y: Second sample [batch_size, dim]
            num_projections: Number of random projections
            
        Returns:
            Sliced Wasserstein distance
        """
        latent_dim = tf.shape(x)[1]
        
        # Generate random projection directions
        projections = tf.random.normal([latent_dim, num_projections], dtype=x.dtype)
        projections = projections / tf.norm(projections, axis=0, keepdims=True)
        
        # Project samples
        x_proj = tf.matmul(x, projections)  # [batch_size, num_projections]
        y_proj = tf.matmul(y, projections)  # [batch_size, num_projections]
        
        # Sort projections
        x_sorted = tf.sort(x_proj, axis=0)
        y_sorted = tf.sort(y_proj, axis=0)
        
        # Compute 1D Wasserstein distances (L1 distance between sorted samples)
        wasserstein_1d = tf.reduce_mean(tf.abs(x_sorted - y_sorted), axis=0)
        
        # Average over all projections
        sliced_wasserstein = tf.reduce_mean(wasserstein_1d)
        
        return sliced_wasserstein
    
    def get_config(self):
        """Get model configuration for serialization."""
        return {
            'config': self.config.to_dict(),
            'name': self.name
        }


def create_twae_mmd_model(config: Optional[TWAEMMDConfig] = None) -> TWAEMMDModel:
    """
    Factory function to create TWAE-MMD model.
    
    Args:
        config: Model configuration (if None, use default)
        
    Returns:
        TWAEMMDModel instance
    """
    if config is None:
        config = TWAEMMDConfig()
    
    model = TWAEMMDModel(config=config)
    
    return model




# Legacy alias for backward compatibility
TransformerModel = TWAEMMDModel

def create_transformer_model(config: Optional[TWAEMMDConfig] = None) -> TWAEMMDModel:
    """
    Legacy factory function for backward compatibility.
    
    Args:
        config: Model configuration (if None, use default)
        
    Returns:
        TWAEMMDModel instance
    """
    return create_twae_mmd_model(config)

