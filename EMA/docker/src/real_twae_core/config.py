"""
TWAE-MMD Model Configuration 

This configuration is specifically tuned for:
- Sequence length: 3-36 amino acids
- Target: >96% classification accuracy
- Model size: ~4M parameters

Author: Reda Mabrouki
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import json
import math
from pathlib import Path


@dataclass
class TWAEMMDConfig:
    """
    Enhanced TWAE-MMD model configuration with latent space support.
    
    This configuration includes all original parameters plus
    latent space and generation parameters for authentic TWAE_MMD.
    """
    
    # ===== Model Architecture  =====
    vocab_size: int = 25                    # 20 amino acids + 5 special tokens
    max_length: int = 37                    # Maximum sequence length (3-36 + padding)
    d_model: int = 256                      # Hidden dimension (128→256, 2x increase)
    num_heads: int = 8                      # Multi-head attention heads
    num_layers: int = 4                     # Transformer layers (3→4, +1 layer)
    d_ff: int = 1024                        # Feed-forward dimension (512→1024, 2x)
    latent_dim: int = 128                   # Latent space dimension (64→128, 2x)
    num_classes: int = 2                    # Binary classification (AMP/non-AMP)
    
    # ===== Latent Space Configuration  =====
    latent_encoder_dims: List[int] = field(default_factory=lambda: [512, 256])
    latent_decoder_dims: List[int] = field(default_factory=lambda: [256, 512])
    latent_activation: str = 'gelu'             # Activation for latent layers
    latent_dropout: float = 0.1                 # Dropout in latent layers
    latent_use_batch_norm: bool = True          # Batch norm in latent space
    latent_use_spectral_norm: bool = False      # Spectral norm for stability
    latent_regularization: str = 'batch_norm'   # 'batch_norm', 'layer_norm', 'none'
    
    # ===== Generation Configuration  =====
    generation_temperature: float = 1.0         # Default generation temperature
    generation_strategy: str = 'gaussian'       # 'gaussian', 'uniform', 'interpolation'
    generation_num_samples: int = 100           # Default number of samples to generate
    sequence_temperature: float = 1.0           # Temperature for sequence sampling
    
    # ===== Latent Space Training  =====
    update_latent_stats: bool = True            # Update latent statistics during training
    latent_stats_momentum: float = 0.99         # Momentum for latent statistics
    latent_warmup_epochs: int = 10              # Epochs before full latent training
    
    # ===== Regularization  =====
    dropout_rate: float = 0.25              # Increased from 0.2 for larger model
    attention_dropout: float = 0.15         # Attention-specific dropout
    layer_dropout: float = 0.1              # Layer-wise dropout (stochastic depth)
    l2_regularization: float = 2e-4         # Increased from 1e-4
    label_smoothing: float = 0.1            # Label smoothing for regularization
    gradient_clip_norm: float = 0.8         # Gradient clipping for stability
    
    # ===== Loss Weights  =====
    classification_weight: float = 1.0      # Primary objective
    mmd_weight: float = 0.35                # Stronger MMD regularization (0.25→0.35)
    wasserstein_weight: float = 0.25        # Increased Wasserstein weight (0.2→0.25)
    reconstruction_weight: float = 0.4      # Enhanced reconstruction (0.3→0.4)
    
    # ===== MMD Configuration =====
    mmd_kernels: List[float] = field(default_factory=lambda: [0.05, 0.2, 0.8, 2.0, 8.0])
    mmd_kernel_type: str = "gaussian"       # Kernel type (gaussian, laplacian, polynomial)
    mmd_adaptive: bool = True               # Adaptive kernel selection
    mmd_num_kernels: int = 5                # Number of kernels for MMD
    
    # ===== Wasserstein Configuration =====
    wasserstein_iterations: int = 15        # Increased iterations (10→15)
    wasserstein_reg: float = 0.08           # Entropy regularization (0.1→0.08)
    wasserstein_type: str = "sinkhorn"      # Sinkhorn divergence
    
    # ===== Training Configuration =====
    use_reconstruction: bool = True         # Enable reconstruction loss
    use_mixed_precision: bool = True        # Mixed precision training
    use_gradient_accumulation: bool = False # Gradient accumulation
    accumulation_steps: int = 1             # Steps for gradient accumulation
    
    # ===== Optimization Parameters =====
    learning_rate: float = 1e-4             # Base learning rate
    warmup_epochs: int = 15                 # Learning rate warmup
    min_learning_rate: float = 1e-6         # Minimum learning rate
    weight_decay: float = 1e-4              # Weight decay
    
    # ===== Training Schedule =====
    epochs: int = 200                       # Maximum training epochs
    batch_size: int = 64                    # Batch size
    validation_split: float = 0.15          # Validation split ratio
    early_stopping_patience: int = 25       # Early stopping patience
    target_accuracy: float = 0.97           # Target accuracy threshold
    
    # ===== Data Configuration =====
    amino_acids: List[str] = field(default_factory=lambda: [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ])
    special_tokens: List[str] = field(default_factory=lambda: [
        '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'
    ])
    
    # ===== Advanced Features =====
    use_layer_scale: bool = True            # Layer scale for training stability
    layer_scale_init: float = 1e-4          # Initial layer scale value
    use_stochastic_depth: bool = True       # Stochastic depth (layer dropout)
    stochastic_depth_rate: float = 0.1      # Stochastic depth probability
    
    # ===== Monitoring and Logging =====
    log_frequency: int = 100                # Logging frequency (steps)
    save_frequency: int = 5                 # Model saving frequency (epochs)
    tensorboard_log_dir: str = "logs/tensorboard"
    checkpoint_dir: str = "results/training/checkpoints"
    
    def __post_init__(self):
        """Initialize derived parameters and validate configuration."""
        # Validate architecture constraints
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_length > 0, "max_length must be positive"
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.num_classes > 0, "num_classes must be positive"
        
        # Validate regularization parameters
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
        assert 0 <= self.attention_dropout <= 1, "attention_dropout must be in [0, 1]"
        assert 0 <= self.layer_dropout <= 1, "layer_dropout must be in [0, 1]"
        
        # Validate latent space configuration
        assert len(self.latent_encoder_dims) > 0, "latent_encoder_dims cannot be empty"
        assert len(self.latent_decoder_dims) > 0, "latent_decoder_dims cannot be empty"
        assert self.latent_regularization in ['batch_norm', 'layer_norm', 'none'], \
            "latent_regularization must be 'batch_norm', 'layer_norm', or 'none'"
        assert 0 <= self.latent_dropout <= 1, "latent_dropout must be in [0, 1]"
        
        # Validate generation parameters
        assert self.generation_temperature > 0, "generation_temperature must be positive"
        assert self.generation_strategy in ['gaussian', 'uniform', 'interpolation'], \
            "generation_strategy must be 'gaussian', 'uniform', or 'interpolation'"
        assert self.generation_num_samples > 0, "generation_num_samples must be positive"
        assert self.sequence_temperature > 0, "sequence_temperature must be positive"
        
        # Validate loss weights
        assert self.classification_weight > 0, "classification_weight must be positive"
        assert self.mmd_weight >= 0, "mmd_weight must be non-negative"
        assert self.wasserstein_weight >= 0, "wasserstein_weight must be non-negative"
        assert self.reconstruction_weight >= 0, "reconstruction_weight must be non-negative"
        
        # Validate training parameters
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.min_learning_rate >= 0, "min_learning_rate must be non-negative"
        assert self.min_learning_rate <= self.learning_rate, \
            "min_learning_rate must be <= learning_rate"
        
        # Validate target accuracy
        assert 0 < self.target_accuracy <= 1, "target_accuracy must be in (0, 1]"
        
        # Create vocabulary mapping
        self.vocab = self.special_tokens + self.amino_acids
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # Validate vocabulary size
        assert len(self.vocab) == self.vocab_size, \
            f"Vocabulary size mismatch: expected {self.vocab_size}, got {len(self.vocab)}"
    
    def get_latent_space_config(self) -> Dict[str, Any]:
        """
        Get configuration for latent space components.
        
        Returns:
            Dict with latent space configuration
        """
        return {
            'latent_dim': self.latent_dim,
            'sequence_length': self.max_length,
            'vocab_size': self.vocab_size,
            'hidden_size': self.d_model,
            'encoder_hidden_dims': self.latent_encoder_dims,
            'decoder_hidden_dims': self.latent_decoder_dims,
            'activation': self.latent_activation,
            'dropout_rate': self.latent_dropout,
            'use_batch_norm': self.latent_use_batch_norm,
            'use_spectral_norm': self.latent_use_spectral_norm,
            'latent_regularization': self.latent_regularization
        }
    
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get configuration for sequence generation.
        
        Returns:
            Dict with generation configuration
        """
        return {
            'temperature': self.generation_temperature,
            'sampling_strategy': self.generation_strategy,
            'num_samples': self.generation_num_samples,
            'sequence_temperature': self.sequence_temperature
        }
    
    def get_parameter_count(self) -> int:
        """
        Estimate total parameter count for the model including latent space.
        
        Returns:
            Estimated number of parameters
        """
        # Token embedding
        token_embed = self.vocab_size * self.d_model
        
        # Positional encoding (learned)
        pos_embed = self.max_length * self.d_model
        
        # Transformer encoder layers
        attention_params_per_layer = (
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            4 * self.d_model                    # Biases
        )
        
        ff_params_per_layer = (
            2 * self.d_model * self.d_ff +     # Two linear layers
            self.d_ff + self.d_model           # Biases
        )
        
        layer_norm_params_per_layer = 2 * 2 * self.d_model  # 2 LayerNorms per layer
        
        encoder_params = self.num_layers * (
            attention_params_per_layer + ff_params_per_layer + layer_norm_params_per_layer
        )
        
        # Global pooling (no parameters)
        
        # Latent space encoder
        latent_encoder_params = 0
        prev_dim = self.d_model
        for dim in self.latent_encoder_dims:
            latent_encoder_params += prev_dim * dim + dim  # weights + bias
            if self.latent_use_batch_norm:
                latent_encoder_params += 2 * dim  # gamma + beta
            prev_dim = dim
        
        # Final latent projection
        latent_encoder_params += prev_dim * self.latent_dim + self.latent_dim
        if self.latent_use_batch_norm:
            latent_encoder_params += 2 * self.latent_dim
        
        # Latent space decoder
        latent_decoder_params = 0
        prev_dim = self.latent_dim
        for dim in self.latent_decoder_dims:
            latent_decoder_params += prev_dim * dim + dim  # weights + bias
            if self.latent_use_batch_norm:
                latent_decoder_params += 2 * dim  # gamma + beta
            prev_dim = dim
        
        # Sequence decoder output
        sequence_output_dim = self.max_length * self.d_model
        latent_decoder_params += prev_dim * sequence_output_dim + sequence_output_dim
        
        # Reconstruction head
        reconstruction_params = self.d_model * self.vocab_size + self.vocab_size
        
        # Classification head
        classifier_params = (
            self.latent_dim * (self.d_model // 2) +
            (self.d_model // 2) * (self.d_model // 4) +
            (self.d_model // 4) * self.num_classes +
            (self.d_model // 2) + (self.d_model // 4) + self.num_classes  # Biases
        )
        
        total = (token_embed + pos_embed + encoder_params + 
                latent_encoder_params + latent_decoder_params + 
                reconstruction_params + classifier_params)
        
        return total
    
    def get_mmd_schedule(self, epoch: float) -> float:
        """
        Get MMD weight for current epoch with warmup schedule.
        
        Args:
            epoch: Current epoch (can be fractional)
            
        Returns:
            MMD weight for the current epoch
        """
        if epoch < self.latent_warmup_epochs:
            # Linear warmup for latent space
            return self.mmd_weight * (epoch / self.latent_warmup_epochs)
        else:
            return self.mmd_weight
    
    def get_learning_rate_schedule(self, epoch: float, total_epochs: int) -> float:
        """
        Get learning rate for current epoch with cosine annealing.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Learning rate for the current epoch
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.learning_rate * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_learning_rate + (self.learning_rate - self.min_learning_rate) * cosine_factor
    
    def get_stochastic_depth_rate(self, layer_idx: int) -> float:
        """
        Get stochastic depth rate for a specific layer.
        
        Args:
            layer_idx: Layer index (0-based)
            
        Returns:
            Stochastic depth rate for the layer
        """
        if not self.use_stochastic_depth:
            return 0.0
        
        # Linear scaling: deeper layers have higher drop rates
        return self.stochastic_depth_rate * layer_idx / (self.num_layers - 1)
    
    def validate_config(self):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # This is called in __post_init__, but can be called manually too
        print(" Configuration validation passed")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'TWAEMMDConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            TWAEMMDConfig instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration
        """
        # Convert to dictionary, excluding computed fields
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in ['vocab', 'token_to_id', 'id_to_token']
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_json(cls, config_path: str) -> 'TWAEMMDConfig':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            TWAEMMDConfig instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, config_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path to save JSON configuration
        """
        # Convert to dictionary, excluding computed fields
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in ['vocab', 'token_to_id', 'id_to_token']
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    def summary(self) -> str:
        """
        Get a summary of the configuration.
        
        Returns:
            Configuration summary string
        """
        param_count = self.get_parameter_count()
        
        summary = f"""
TWAE-MMD Configuration Summary (with Latent Space)
==================================================

Model Architecture:
  - Parameters: ~{param_count:,} ({param_count/1e6:.1f}M)
  - d_model: {self.d_model}
  - num_layers: {self.num_layers}
  - num_heads: {self.num_heads}
  - latent_dim: {self.latent_dim}
  - vocab_size: {self.vocab_size}
  - max_length: {self.max_length}

Latent Space Configuration:
  - Encoder dims: {self.latent_encoder_dims}
  - Decoder dims: {self.latent_decoder_dims}
  - Activation: {self.latent_activation}
  - Regularization: {self.latent_regularization}
  - Batch norm: {self.latent_use_batch_norm}
  - Spectral norm: {self.latent_use_spectral_norm}

Generation Configuration:
  - Strategy: {self.generation_strategy}
  - Temperature: {self.generation_temperature}
  - Sequence temperature: {self.sequence_temperature}
  - Default samples: {self.generation_num_samples}

Training Configuration:
  - Target accuracy: {self.target_accuracy:.1%}
  - Epochs: {self.epochs}
  - Batch size: {self.batch_size}
  - Learning rate: {self.learning_rate}
  - Warmup epochs: {self.warmup_epochs}
  - Latent warmup: {self.latent_warmup_epochs}

Regularization:
  - Dropout rate: {self.dropout_rate}
  - Latent dropout: {self.latent_dropout}
  - L2 regularization: {self.l2_regularization}
  - Label smoothing: {self.label_smoothing}
  - Gradient clipping: {self.gradient_clip_norm}

Loss Weights:
  - Classification: {self.classification_weight}
  - MMD: {self.mmd_weight}
  - Wasserstein: {self.wasserstein_weight}
  - Reconstruction: {self.reconstruction_weight}

Advanced Features:
  - Mixed precision: {self.use_mixed_precision}
  - Stochastic depth: {self.use_stochastic_depth}
  - Layer scale: {self.use_layer_scale}
  - MMD adaptive: {self.mmd_adaptive}
  - Reconstruction: {self.use_reconstruction}
        """
        return summary.strip()
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return self.summary()


# Default configuration for quick access
DEFAULT_CONFIG = TWAEMMDConfig()

# Configuration presets for different scenarios
CONFIGS = {
    "default": TWAEMMDConfig(),
    
    "small": TWAEMMDConfig(
        d_model=128,
        num_layers=3,
        d_ff=512,
        latent_dim=64,
        latent_encoder_dims=[256, 128],
        latent_decoder_dims=[128, 256],
        dropout_rate=0.2,
        mmd_weight=0.25,
    ),
    
    "large": TWAEMMDConfig(
        d_model=512,
        num_layers=6,
        d_ff=2048,
        latent_dim=256,
        latent_encoder_dims=[1024, 512],
        latent_decoder_dims=[512, 1024],
        dropout_rate=0.3,
        mmd_weight=0.4,
    ),
    
    "fast_training": TWAEMMDConfig(
        epochs=100,
        batch_size=128,
        learning_rate=2e-4,
        warmup_epochs=10,
        latent_warmup_epochs=5,
    ),
    
    "high_accuracy": TWAEMMDConfig(
        target_accuracy=0.98,
        epochs=100,
        early_stopping_patience=50,
        mmd_weight=0.4,
        reconstruction_weight=0.5,
        latent_warmup_epochs=15,
    ),
    
    "generation_focused": TWAEMMDConfig(
        reconstruction_weight=0.6,
        mmd_weight=0.4,
        wasserstein_weight=0.3,
        generation_temperature=0.8,
        latent_regularization='layer_norm',
        latent_use_batch_norm=True,
    ),
}


def get_config(name: str = "default") -> TWAEMMDConfig:
    """
    Get a predefined configuration.
    
    Args:
        name: Configuration name
        
    Returns:
        TWAEMMDConfig instance
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown configuration: {name}. Available: {list(CONFIGS.keys())}")
    
    return CONFIGS[name]


