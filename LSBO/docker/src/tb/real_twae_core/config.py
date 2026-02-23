"""
TWAE-MMD Model Configuration with Transformer Decoder
+4M Parameters with Transformer-based Reconstruction (No MLP)

This configuration is specifically tuned for:
- Sequence length: 3-36 amino acids
- Target: >97% classification accuracy
- Model size: +4M parameters
- Transformer Decoder for autoregressive reconstruction
- Authentic TWAE_MMD with latent space for high-quality AMP generation

Author: Reda Mabrouki
Updated: Transformer Decoder Only (No MLP)
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
    TWAE-MMD model configuration with Transformer Decoder.
    
    This configuration uses Transformer Decoder for reconstruction
    instead of simple MLP, providing better sequence modeling.
    """
    
    # ===== Model Architecture =====
    vocab_size: int = 25                    # 20 amino acids + 5 special tokens
    max_length: int = 37                    # Maximum sequence length (3-36 + padding)
    d_model: int = 256                      # Hidden dimension
    num_heads: int = 8                      # Multi-head attention heads
    
    # Encoder configuration
    encoder_num_layers: int = 4             # Transformer encoder layers
    
    # Decoder configuration (Transformer, not MLP!)
    decoder_num_layers: int = 4             # Transformer decoder layers
    
    d_ff: int = 1024                        # Feed-forward dimension
    latent_dim: int = 128                   # Latent space dimension (INCREASED from 128 for better reconstruction)
    num_classes: int = 2                    # Binary classification (AMP/non-AMP)
    
    # ===== Latent Space Configuration =====
    # Note: Latent encoder/decoder are for latent vector processing,
    # NOT for sequence reconstruction (that's done by Transformer Decoder)
    latent_encoder_dims: List[int] = field(default_factory=lambda: [512, 256])
    latent_activation: str = 'gelu'             # Activation for latent layers
    latent_dropout: float = 0.1                 # Dropout in latent layers
    latent_use_batch_norm: bool = True          # Batch norm in latent space
    latent_use_spectral_norm: bool = False      # Spectral norm for stability
    latent_regularization: str = 'batch_norm'   # 'batch_norm', 'layer_norm', 'none'
    
    # ===== Generation Configuration =====
    generation_temperature: float = 1.0         # Default generation temperature
    generation_strategy: str = 'learned'        # 'gaussian', 'uniform', 'learned'
    generation_num_samples: int = 100           # Default number of samples to generate
    sequence_temperature: float = 1.0           # Temperature for sequence sampling
    
    # ===== Latent Space Training =====
    update_latent_stats: bool = True            # Update latent statistics during training
    latent_stats_momentum: float = 0.99         # Momentum for latent statistics (EMA)
    latent_warmup_epochs: int = 10              # Epochs before full latent training
    
    # ===== Regularization =====
    dropout_rate: float = 0.25              # General dropout rate
    attention_dropout: float = 0.15         # Attention-specific dropout
    layer_dropout: float = 0.1              # Layer-wise dropout (stochastic depth)
    l2_regularization: float = 2e-4         # L2 weight regularization
    label_smoothing: float = 0.1            # Label smoothing for classification
    gradient_clip_norm: float = 0.8         # Gradient clipping for stability
    
    # ===== Loss Weights (REBALANCED - Fixed reconstruction loss increasing) =====
    classification_weight: float = 0.5      # Classification objective
    reconstruction_weight: float = 0.7      # Reconstruction quality (INCREASED from 0.25)
    mmd_weight: float = 0.2                 # MMD distribution matching (DECREASED from 0.4)
    wasserstein_weight: float = 0.1         # Wasserstein regularization (DECREASED from 0.3)
    
    # ===== MMD Configuration =====
    mmd_kernels: List[float] = field(default_factory=lambda: [0.05, 0.2, 0.8, 2.0, 8.0])
    mmd_kernel_type: str = "mixed"          # Kernel type (gaussian, imq, mixed)
    mmd_adaptive: bool = True               # Adaptive kernel selection
    mmd_num_kernels: int = 5                # Number of kernels for MMD
    
    # ===== Wasserstein Configuration =====
    wasserstein_iterations: int = 15        # Sinkhorn iterations
    wasserstein_reg: float = 0.05           # Entropy regularization (epsilon)
    wasserstein_type: str = "sinkhorn"      # Sinkhorn divergence
    
    # ===== Training Configuration =====
    use_mixed_precision: bool = True        # Mixed precision training
    use_gradient_accumulation: bool = False # Gradient accumulation
    accumulation_steps: int = 1             # Steps for gradient accumulation
    
    # ===== Optimization Parameters =====
    learning_rate: float = 1e-5             # Base learning rate (very conservative for stability)
    warmup_epochs: int = 15                 # Learning rate warmup
    min_learning_rate: float = 1e-6         # Minimum learning rate
    weight_decay: float = 1e-4              # Weight decay
    
    # ===== Training Schedule =====
    max_epochs: int = 200                   # Maximum training epochs
    batch_size: int = 64                    # Batch size (may need to reduce to 32 or 24)
    validation_split: float = 0.15          # Validation split ratio
    early_stopping_patience: int = 25       # Early stopping patience
    target_accuracy: float = 0.97           # Target accuracy threshold
    
    # ===== Data Configuration =====
    amino_acids: List[str] = field(default_factory=lambda: [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ])
    special_tokens: List[str] = field(default_factory=lambda: [
        '[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]'
    ])
    
    # Token IDs (must match special_tokens order)
    cls_token_id: int = 0
    sep_token_id: int = 1
    pad_token_id: int = 2
    mask_token_id: int = 3
    unk_token_id: int = 4
    
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
        
        assert self.encoder_num_layers > 0, "encoder_num_layers must be positive"
        assert self.decoder_num_layers > 0, "decoder_num_layers must be positive"
        
        # Validate regularization parameters
        assert 0 <= self.dropout_rate <= 1, "dropout_rate must be in [0, 1]"
        assert 0 <= self.attention_dropout <= 1, "attention_dropout must be in [0, 1]"
        assert 0 <= self.layer_dropout <= 1, "layer_dropout must be in [0, 1]"
        
        # Validate latent space configuration
        assert len(self.latent_encoder_dims) > 0, "latent_encoder_dims cannot be empty"
        assert self.latent_regularization in ['batch_norm', 'layer_norm', 'none'], \
            "latent_regularization must be 'batch_norm', 'layer_norm', or 'none'"
        assert 0 <= self.latent_dropout <= 1, "latent_dropout must be in [0, 1]"
        
        # Validate generation parameters
        assert self.generation_temperature > 0, "generation_temperature must be positive"
        assert self.generation_strategy in ['gaussian', 'uniform', 'learned'], \
            "generation_strategy must be 'gaussian', 'uniform', or 'learned'"
        assert self.generation_num_samples > 0, "generation_num_samples must be positive"
        
        # Validate loss weights
        assert self.classification_weight >= 0, "classification_weight must be non-negative"
        assert self.reconstruction_weight >= 0, "reconstruction_weight must be non-negative"
        assert self.mmd_weight >= 0, "mmd_weight must be non-negative"
        assert self.wasserstein_weight >= 0, "wasserstein_weight must be non-negative"
        
        # Validate training parameters
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.max_epochs > 0, "max_epochs must be positive"
        assert 0 < self.validation_split < 1, "validation_split must be in (0, 1)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            # Architecture
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'encoder_num_layers': self.encoder_num_layers,
            'decoder_num_layers': self.decoder_num_layers,
            'd_ff': self.d_ff,
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
            
            # Latent space
            'latent_encoder_dims': self.latent_encoder_dims,
            'latent_activation': self.latent_activation,
            'latent_dropout': self.latent_dropout,
            'latent_use_batch_norm': self.latent_use_batch_norm,
            'latent_regularization': self.latent_regularization,
            
            # Generation
            'generation_temperature': self.generation_temperature,
            'generation_strategy': self.generation_strategy,
            'generation_num_samples': self.generation_num_samples,
            
            # Regularization
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'layer_dropout': self.layer_dropout,
            'l2_regularization': self.l2_regularization,
            'label_smoothing': self.label_smoothing,
            'gradient_clip_norm': self.gradient_clip_norm,
            
            # Loss weights
            'classification_weight': self.classification_weight,
            'reconstruction_weight': self.reconstruction_weight,
            'mmd_weight': self.mmd_weight,
            'wasserstein_weight': self.wasserstein_weight,
            
            # Training
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
        }
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: str) -> 'TWAEMMDConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls(**config_dict)
    
    def estimate_parameters(self) -> int:
        """
        Estimate total number of model parameters.
        
        Returns:
            Estimated parameter count
        """
        # Encoder parameters
        encoder_params = 0
        
        # Token embedding
        encoder_params += self.vocab_size * self.d_model
        
        # Positional encoding (learned)
        encoder_params += self.max_length * self.d_model
        
        # Encoder layers
        for _ in range(self.encoder_num_layers):
            # Multi-head attention
            encoder_params += 4 * self.d_model * self.d_model  # Q, K, V, O projections
            # Feed-forward
            encoder_params += 2 * self.d_model * self.d_ff  # Two linear layers
            # Layer norms (2 per layer)
            encoder_params += 4 * self.d_model  # gamma + beta for each norm
        
        # Decoder parameters (Transformer Decoder)
        decoder_params = 0
        
        # Token embedding (shared or separate)
        decoder_params += self.vocab_size * self.d_model
        
        # Positional encoding
        decoder_params += self.max_length * self.d_model
        
        # Latent projection (latent_dim -> d_model)
        decoder_params += self.latent_dim * self.d_model + self.d_model
        
        # Decoder layers
        for _ in range(self.decoder_num_layers):
            # Self-attention
            decoder_params += 4 * self.d_model * self.d_model
            # Cross-attention to latent
            decoder_params += 4 * self.d_model * self.d_model
            # Feed-forward
            decoder_params += 2 * self.d_model * self.d_ff
            # Layer norms (3 per layer: after self-attn, cross-attn, ffn)
            decoder_params += 6 * self.d_model
        
        # Output projection (d_model -> vocab_size)
        decoder_params += self.d_model * self.vocab_size
        
        # Latent space parameters
        latent_params = 0
        
        # Latent encoder (d_model -> latent_dim)
        prev_dim = self.d_model
        for dim in self.latent_encoder_dims:
            latent_params += prev_dim * dim + dim  # weights + bias
            if self.latent_use_batch_norm:
                latent_params += 2 * dim  # gamma + beta
            prev_dim = dim
        
        # Final projection to latent_dim
        latent_params += prev_dim * self.latent_dim + self.latent_dim
        
        # Classification head
        classification_params = self.latent_dim * self.num_classes + self.num_classes
        
        # Total parameters
        total_params = (
            encoder_params + 
            decoder_params + 
            latent_params + 
            classification_params
        )
        
        return total_params
    
    def __str__(self) -> str:
        """String representation of configuration."""
        estimated_params = self.estimate_parameters()
        
        return f"""
TWAE-MMD Configuration (Transformer Decoder)
{'=' * 60}
Architecture:
  - Vocab size: {self.vocab_size}
  - Max length: {self.max_length}
  - Model dimension: {self.d_model}
  - Attention heads: {self.num_heads}
  - Encoder layers: {self.encoder_num_layers}
  - Decoder layers: {self.decoder_num_layers} (Transformer, not MLP!)
  - Feed-forward dim: {self.d_ff}
  - Latent dimension: {self.latent_dim}
  - Num classes: {self.num_classes}
  
Latent Space:
  - Encoder dims: {self.latent_encoder_dims}
  - Activation: {self.latent_activation}
  - Dropout: {self.latent_dropout}
  - Batch norm: {self.latent_use_batch_norm}
  - Regularization: {self.latent_regularization}
  
Generation:
  - Temperature: {self.generation_temperature}
  - Strategy: {self.generation_strategy}
  - Num samples: {self.generation_num_samples}
  
Regularization:
  - Dropout: {self.dropout_rate}
  - Attention dropout: {self.attention_dropout}
  - Layer dropout: {self.layer_dropout}
  - L2 reg: {self.l2_regularization}
  - Label smoothing: {self.label_smoothing}
  - Gradient clip: {self.gradient_clip_norm}
  
Loss Weights:
  - Classification: {self.classification_weight}
  - Reconstruction: {self.reconstruction_weight}
  - MMD: {self.mmd_weight}
  - Wasserstein: {self.wasserstein_weight}
  
Training:
  - Learning rate: {self.learning_rate}
  - Batch size: {self.batch_size}
  - Max epochs: {self.max_epochs}
  - Validation split: {self.validation_split}
  - Early stopping: {self.early_stopping_patience} epochs
  - Target accuracy: {self.target_accuracy:.1%}
  
Estimated Parameters: {estimated_params:,} (~{estimated_params/1e6:.2f}M)
{'=' * 60}
"""


# Factory functions for common configurations

def create_default_config() -> TWAEMMDConfig:
    """Create default TWAE-MMD configuration with Transformer Decoder."""
    return TWAEMMDConfig()


def create_small_config() -> TWAEMMDConfig:
    """Create smaller configuration for testing."""
    return TWAEMMDConfig(
        d_model=128,
        num_heads=4,
        encoder_num_layers=2,
        decoder_num_layers=2,
        d_ff=512,
        latent_dim=64,
        latent_encoder_dims=[256, 128],
        batch_size=32
    )


def create_large_config() -> TWAEMMDConfig:
    """Create larger configuration for better performance."""
    return TWAEMMDConfig(
        d_model=512,
        num_heads=16,
        encoder_num_layers=6,
        decoder_num_layers=6,
        d_ff=2048,
        latent_dim=128,
        latent_encoder_dims=[1024, 512],
        batch_size=32  # Smaller batch due to model size
    )




# ===== Configuration Presets =====

# Default configuration instance
DEFAULT_CONFIG = TWAEMMDConfig()

# Configuration presets for different scenarios
CONFIGS = {
    "default": TWAEMMDConfig(),
    
    "small": TWAEMMDConfig(
        d_model=128,
        encoder_num_layers=2,
        decoder_num_layers=2,
        d_ff=512,
        latent_dim=64,
        latent_encoder_dims=[256, 128],
        dropout_rate=0.2,
        mmd_weight=0.25,
        batch_size=32
    ),
    
    "large": TWAEMMDConfig(
        d_model=512,
        encoder_num_layers=6,
        decoder_num_layers=6,
        d_ff=2048,
        latent_dim=128,
        latent_encoder_dims=[1024, 512],
        dropout_rate=0.3,
        mmd_weight=0.4,
        batch_size=32
    ),
    
    "fast_training": TWAEMMDConfig(
        max_epochs=20,
        batch_size=128,
        learning_rate=2e-4,
        warmup_epochs=10,
        latent_warmup_epochs=5,
    ),
    
    "high_accuracy": TWAEMMDConfig(
        latent_dim=128,  # CRITICAL: Must explicitly set to override default!
        target_accuracy=0.98,
        max_epochs=150,
        early_stopping_patience=50,
        mmd_weight=0.3,  # FIXED: Use balanced weight (was 0.4)
        wasserstein_weight=0.2,  # FIXED: Use balanced weight (was missing, defaults to 0.3)
        reconstruction_weight=1.0,  # FIXED: Increased from 0.5 to maintain importance throughout training
        latent_warmup_epochs=15,
    ),
    
    "generation_focused": TWAEMMDConfig(
        latent_dim=128,  # CRITICAL: Must explicitly set to override default!
        reconstruction_weight=1.0,  # FIXED: Increased from 0.6 to maintain importance
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

