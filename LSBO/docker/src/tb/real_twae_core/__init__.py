"""
Production TWAE Core Module 
Complete TWAE_MMD architecture with Transformer Decoder (no MLP)

This module provides the complete production-ready TWAE_MMD components:
- Transformer encoder and decoder
- Latent space encoder (no MLP decoder)
- Attention mechanisms
- Positional encoding
- Layer implementations
- Model configuration
- COMPLETE LOSS FUNCTIONS: Classification, Reconstruction, MMD, Wasserstein
- High-quality AMP generation capabilities

Author: Reda Mabrouki
Updated: 2025 - Transformer Decoder Only (No MLP)
"""

__version__ = "3.1.0"
__author__ = "Reda Mabrouki"
__email__ = "reda.mabrouki@bioinfsys.uni-gissen.de"
__license__ = ""

# Core imports
from .config import (
    TWAEMMDConfig,
    DEFAULT_CONFIG,
    CONFIGS,
    get_config
)

from .model import (
    TWAEMMDModel,
    TransformerModel,  # Legacy alias
    create_transformer_model,
    create_twae_mmd_model
)

# Latent space imports (NO LatentSpaceDecoder - using Transformer Decoder instead)
from .latent_space import (
    LatentSpaceManager,
    LatentSpaceEncoder,
    LatentSpaceConfig,
    create_latent_space_manager
)

# Component imports
from .encoder import (
    TransformerEncoder
)

from .decoder import (
    TransformerDecoder
)

from .attention import (
    SelfAttention,
    CrossAttention,
    CausalSelfAttention,
    MultiHeadAttention
)

from .layers import (
    FeedForwardNetwork,
    TransformerBlock,
    DecoderBlock,
    StochasticDepth,
    create_transformer_block
)

from .positional_encoding import (
    PositionalEncoding,
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RelativePositionalEncoding,
    create_positional_encoding
)

# Version information
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "description": "Production TWAE_MMD Architecture - Transformer Decoder Only",
    "target_accuracy": ">97%",
    "model_size": "~4.7M parameters",
    "framework": "TensorFlow 2.13+",
    "python_version": "3.8+",
    "status": "PRODUCTION READY - TRANSFORMER DECODER",
    "features": [
        "Transformer Encoder (BERT-like)",
        "Transformer Decoder (GPT-like)",
        "Cross-Attention to Latent Vector",
        "Autoregressive Generation",
        "High-Quality AMP Generation",
        "Sequence Interpolation",
        "Multiple Sampling Strategies",
        "Generation Quality Analysis",
        "Authentic TWAE_MMD Training",
        "COMPLETE Loss Functions Integration",
        "MMD Loss Implementation",
        "Wasserstein Loss Implementation",
        "Production-Ready Code Only"
    ],
    "loss_functions": [
        "Classification Loss (compute_classification_loss)",
        "Reconstruction Loss (compute_reconstruction_loss)", 
        "MMD Loss (compute_mmd_loss)",
        "Wasserstein Loss (compute_wasserstein_loss)"
    ],
    "components": [
        "Transformer Encoder",
        "Transformer Decoder (GPT-like)", 
        "Latent Space Manager",
        "Latent Space Encoder (NO MLP Decoder)",
        "Multi-Head Attention",
        "Cross-Attention to Latent",
        "Positional Encoding",
        "Feed-Forward Networks",
        "Layer Normalization",
        "Stochastic Depth",
        "Layer Scaling",
        "Generation Capabilities",
        "Quality Analysis Tools",
        "Complete TWAE-MMD Loss Suite"
    ],
    "architecture_changes": {
        "removed_mlp_decoder": True,
        "using_transformer_decoder": True,
        "cross_attention_to_latent": True,
        "autoregressive_generation": True
    }
}

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "VERSION_INFO",
    
    # Configuration
    "TWAEMMDConfig",
    "DEFAULT_CONFIG", 
    "CONFIGS",
    "get_config",
    
    # Main model
    "TWAEMMDModel",
    "TransformerModel",
    "create_transformer_model",
    "create_twae_mmd_model",
    
    # Latent space (NO LatentSpaceDecoder, NO LatentSpaceAnalyzer)
    "LatentSpaceManager",
    "LatentSpaceEncoder",
    "LatentSpaceConfig",
    "create_latent_space_manager",
    
    # Core components
    "TransformerEncoder",
    "TransformerDecoder",
    
    # Attention mechanisms
    "SelfAttention",
    "CrossAttention", 
    "CausalSelfAttention",
    "MultiHeadAttention",
    
    # Layers
    "FeedForwardNetwork",
    "TransformerBlock",
    "DecoderBlock",
    "StochasticDepth",
    "create_transformer_block",
    
    # Positional encoding
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RelativePositionalEncoding",
    "create_positional_encoding"
]

# Configuration presets
PRESET_CONFIGS = {
    'production': 'Production configuration for real training (>97% accuracy)',
    'high_accuracy': 'Configuration optimized for maximum accuracy',
    'fast_training': 'Configuration optimized for fast training',
    'balanced': 'Balanced configuration for general use',
    'debug': 'Configuration for debugging and development'
}

def get_available_configs():
    """Get list of available configuration presets."""
    return list(PRESET_CONFIGS.keys())

def print_production_status():
    """Print production status and usage guide."""
    print("=" * 60)
    print("PRODUCTION TWAE-MMD Framework v3.1.0")
    print("Transformer Decoder Only (No MLP)")
    print("=" * 60)
    print()

    print("✅ ARCHITECTURE:")
    print("   • Transformer Encoder (BERT-like)")
    print("   • Transformer Decoder (GPT-like)")
    print("   • Cross-Attention to Latent Vector")
    print("   • Autoregressive Generation")
    print("   • NO MLP Decoder (removed)")
    print()

    print("✅ COMPLETE LOSS FUNCTIONS:")
    print("   • Classification Loss")
    print("   • Reconstruction Loss")
    print("   • MMD Loss")
    print("   • Wasserstein Loss")
    print()

    print("📝 USAGE:")
    print("  from real_twae_core import create_twae_mmd_model, get_config")
    print("  config = get_config('default')")
    print("  model = create_twae_mmd_model(config)")
    print("  # Ready for real training with authentic datasets!")
    print()

    print("⚙️  AVAILABLE CONFIGURATIONS:")
    for config_name, description in PRESET_CONFIGS.items():
        print(f"   • {config_name}: {description}")
    print()
    print("=" * 60)

# Initialize production module
if __name__ == "__main__":
    print(f"Production TWAE-MMD Framework v{__version__}")
    print("Transformer Decoder Architecture")
    print()
    print_production_status()

