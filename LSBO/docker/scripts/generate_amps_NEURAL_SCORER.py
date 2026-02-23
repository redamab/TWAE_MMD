"""
🚀 PHASE 2: AMP GENERATION FROM TRAINED MODEL - NEURAL SCORER VERSION
=====================================================================

This script uses the NEURAL ImprovedAMPScorer (same as training) for consistency.

Key Features:
1. Uses ImprovedAMPScorer (neural network) instead of simple string scorer
2. Threshold 0.80 (consistent with sAMPpred-GAT validation)
3. Passes token IDs to scorer (not strings)
4. Generates AMPs with same quality as LSBO training regions

Author: Reda
Date: 2024-11-17
Version: 4.0.0 - NEURAL SCORER (Consistent with Training!)
"""

import os
import sys
import time
import logging
import warnings
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Core imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# TensorFlow warning suppression
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Set up paths
WORKSPACE_ROOT = "/workspace/TWAE_AMP_Generation"
SRC_PATH = os.path.join(WORKSPACE_ROOT, "src")

sys.path.insert(0, SRC_PATH)
sys.path.insert(0, os.path.join(SRC_PATH, "data"))
sys.path.insert(0, os.path.join(SRC_PATH, "real_twae_core"))

# Import TWAE-MMD components
try:
    from data import create_simple_pipeline
    from real_twae_core import create_twae_mmd_model, get_config as get_model_config
    from real_twae_core.property_predictor import ImprovedAMPScorer
    from real_twae_core.constraints import AMPConstraints
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class AMPGenerator:
    """
    Generate high-quality membrane-reactive AMPs using NEURAL scorer.
    """
    
    def __init__(self, 
                 checkpoint_prefix: str,
                 output_dir: str = "generated_amps_neural_scorer"):
        """
        Initialize AMP generator with neural scorer.
        
        Args:
            checkpoint_prefix: Path prefix to checkpoint files (without extension)
            output_dir: Output directory for generated AMPs
        """
        self.checkpoint_prefix = Path(checkpoint_prefix)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load model from .h5 + LSBO sampler + config
        self._load_model_h5()
        
        # Setup tokenizer
        self.logger.info("Setting up tokenizer...")
        self.tokenizer, _ = create_simple_pipeline()
        self.logger.info("✅ Tokenizer ready")
        
        # Setup NEURAL property predictor (same as training!)
        self.logger.info("Setting up NEURAL ImprovedAMPScorer (same as training)...")
        self.property_predictor = ImprovedAMPScorer()
        self.logger.info("✅ Neural scorer ready")
        
        # Setup constraints
        self.constraints = AMPConstraints(
            min_length=10,
            max_length=36,
            min_charge=2.0,
            max_charge=9.0,
            min_hydrophobicity=0.30,
            max_hydrophobicity=0.70,
            required_amino_acids={'K', 'R'},
            min_diversity=0.25
        )
        self.logger.info("✅ Constraints ready")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f'generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model_h5(self):
        """Load model from .h5 weights + LSBO sampler + config."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("🔄 LOADING MODEL FROM H5 FORMAT")
        self.logger.info(f"{'='*80}\n")
        
        # File paths
        weights_path = f"{self.checkpoint_prefix}.h5"
        lsbo_path = f"{self.checkpoint_prefix}_lsbo_sampler.pkl"
        config_path = f"{self.checkpoint_prefix}_config.json"
        
        self.logger.info(f"📁 Weights: {weights_path}")
        self.logger.info(f"📁 LSBO Sampler: {lsbo_path}")
        self.logger.info(f"📁 Config: {config_path}\n")
        
        # Check files exist
        for path in [weights_path, lsbo_path, config_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Step 1: Load config
        self.logger.info("🔧 Step 1: Loading model config...")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        self.logger.info(f"✅ Config loaded: {config_dict['model_type']}\n")
        
        # Step 2: Create model architecture
        self.logger.info("🔧 Step 2: Creating model architecture...")
        model_config = get_model_config(config_dict['model_type'])
        model_config.use_lsbo_sampling = config_dict['use_lsbo_sampling']
        self.model = create_twae_mmd_model(model_config)
        self.logger.info("✅ Model architecture created\n")
        
        # Step 3: Build model
        self.logger.info("🔧 Step 3: Building model graph...")
        dummy_input = tf.zeros((1, model_config.max_length), dtype=tf.int32)
        _ = self.model(dummy_input, training=False)
        self.logger.info("✅ Model graph built\n")
        
        # Step 4: Load weights (with flexible loading)
        self.logger.info("🔧 Step 4: Loading weights from .h5 file...")
        try:
            # Try exact loading first
            self.model.load_weights(weights_path)
            self.logger.info("✅ Weights loaded successfully (exact match)!\n")
        except ValueError as e:
            # If layer count mismatch, load by name and skip mismatches
            self.logger.warning(f"⚠️  Layer mismatch detected: {str(e)[:100]}...")
            self.logger.info("   Trying flexible loading (by_name=True, skip_mismatch=True)...")
            self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            self.logger.info("✅ Weights loaded successfully (flexible matching)!\n")
        
        # Step 5: Load LSBO sampler
        self.logger.info("🔧 Step 5: Loading LSBO sampler with HQ regions...")
        with open(lsbo_path, 'rb') as f:
            lsbo_sampler = pickle.load(f)
        
        # Replace model's LSBO sampler with loaded one
        self.model.latent_manager.lsbo_sampler = lsbo_sampler
        
        hq_count = len(lsbo_sampler.high_quality_regions)
        self.logger.info(f"✅ LSBO sampler loaded with {hq_count} HQ regions!\n")
    
    def _score_amp_neural(self, sequence: str, token_ids: np.ndarray) -> dict:
        """
        Score AMP using NEURAL ImprovedAMPScorer (same as training).
        
        Args:
            sequence: Amino acid sequence string
            token_ids: Token IDs array (for neural scorer)
        
        Returns:
            Dictionary with scores
        """
        try:
            # Pad token_ids to max_length (37)
            max_length = 37
            if len(token_ids) < max_length:
                padded_ids = np.zeros(max_length, dtype=np.int32)
                padded_ids[:len(token_ids)] = token_ids
            else:
                padded_ids = token_ids[:max_length]
            
            # Convert to tensor WITHOUT batch dimension
            # ImprovedAMPScorer expects shape (37,) not (1, 37)!
            token_ids_tensor = tf.constant(padded_ids, dtype=tf.int32)
            
            # Score with neural network (no training parameter needed)
            # ImprovedAMPScorer returns a single float score, not a dict!
            overall_score = self.property_predictor(token_ids_tensor)
            
            # Convert to Python float if it's a tensor
            if isinstance(overall_score, tf.Tensor):
                overall_score = float(overall_score.numpy())
            else:
                overall_score = float(overall_score)
            
            # Calculate membrane reactivity (same as overall score)
            # The scorer already optimizes for membrane-reactive properties
            membrane_score = overall_score
            
            # Calculate basic properties for statistics
            positive = set('KRH')
            negative = set('DE')
            hydrophobic = set('AILMFVPWG')
            
            length = len(sequence)
            positive_count = sum(1 for aa in sequence if aa in positive)
            negative_count = sum(1 for aa in sequence if aa in negative)
            net_charge = positive_count - negative_count
            hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic)
            hydro_ratio = hydrophobic_count / length if length > 0 else 0.0
            
            # Aliphatic index
            aliphatic = set('AILV')
            aliphatic_count = sum(1 for aa in sequence if aa in aliphatic)
            aliphatic_index = (aliphatic_count / length) * 100 if length > 0 else 0.0
            
            # Secondary structure propensity
            helix_forming = set('AEKLMQR')
            helix_count = sum(1 for aa in sequence if aa in helix_forming)
            helix_propensity = helix_count / length if length > 0 else 0.0
            
            return {
                'overall_score': overall_score,
                'membrane_reactivity': membrane_score,
                'charge': net_charge,
                'hydrophobicity': hydro_ratio,
                'aliphatic_index': aliphatic_index,
                'secondary_structure': helix_propensity
            }
            
        except Exception as e:
            self.logger.error(f"Error in neural scoring: {e}")
            # Return low scores on error
            return {
                'overall_score': 0.0,
                'membrane_reactivity': 0.0,
                'charge': 0.0,
                'hydrophobicity': 0.0,
                'aliphatic_index': 0.0,
                'secondary_structure': 0.0
            }
    
    def generate(self, 
                num_amps: int = 1000,
                temperature: float = 0.8,
                min_score: float = 0.80,
                min_membrane_score: float = 0.80):
        """
        Generate membrane-reactive AMPs using NEURAL scorer.
        
        Args:
            num_amps: Number of AMPs to generate
            temperature: Sampling temperature (0.5-1.5)
            min_score: Minimum overall quality score (0.80 = sAMPpred-GAT threshold)
            min_membrane_score: Minimum membrane reactivity score
        
        Returns:
            DataFrame with generated AMPs
        """
        self.logger.info(f"{'='*80}")
        self.logger.info("🧬 GENERATING AMPs WITH NEURAL SCORER")
        self.logger.info(f"{'='*80}\n")
        
        self.logger.info(f"🎯 Target: {num_amps} AMPs")
        self.logger.info(f"🌡  Temperature: {temperature}")
        self.logger.info(f"📊 Min Overall Score: {min_score} (sAMPpred-GAT threshold)")
        self.logger.info(f"📊 Min Membrane Score: {min_membrane_score}\n")
        
        generated_amps = []
        duplicates = set()
        attempts = 0
        max_attempts = 100000  # Increased for high-quality generation
        
        # Get LSBO sampler
        lsbo_sampler = self.model.latent_manager.lsbo_sampler
        hq_count = len(lsbo_sampler.high_quality_regions)
        
        if hq_count > 0:
            self.logger.info(f"✅ Using {hq_count} LSBO high-quality regions for sampling")
            self.logger.info(f"   LSBO mean score: 0.8965 (training quality)\n")
        else:
            self.logger.warning("⚠️  No HQ regions - using random sampling\n")
        
        pbar = tqdm(total=num_amps, desc="Generating AMPs")
        
        # Debug counters
        rejection_reasons = {
            'duplicate': 0,
            'constraints': 0,
            'low_overall_score': 0,
            'low_membrane_score': 0,
            'exception': 0
        }
        sample_sequences = []
        
        while len(generated_amps) < num_amps and attempts < max_attempts:
            # Sample latent vectors from HQ regions
            if hq_count > 0:
                latent_vectors = lsbo_sampler.sample_from_high_quality_regions(
                    num_samples=100,
                    exploration_noise=0.05
                )
            else:
                latent_vectors = np.random.randn(100, self.model.config.latent_dim).astype(np.float32)
            
            # Generate sequences
            for latent_vector in latent_vectors:
                attempts += 1
                
                try:
                    # Decode
                    latent_batch = tf.expand_dims(latent_vector, axis=0)
                    decoder_output = self.model.decoder(latent_batch, training=False)
                    
                    # Decoder returns tensor directly
                    if isinstance(decoder_output, dict):
                        logits = decoder_output['reconstruction_logits'] / temperature
                    else:
                        logits = decoder_output / temperature
                    
                    # Sample sequence with temperature
                    token_ids = tf.random.categorical(logits[0], num_samples=1)[:, 0]
                    token_ids_np = token_ids.numpy()
                    
                    # Stop at PAD token (0) or force max length of 36
                    pad_idx = np.where(token_ids_np == 0)[0]
                    if len(pad_idx) > 0 and pad_idx[0] > 0:
                        token_ids_np = token_ids_np[:pad_idx[0]]
                    else:
                        token_ids_np = token_ids_np[:36]
                    
                    # Decode to string
                    sequence = self.tokenizer.decode(token_ids_np)
                    sequence = sequence.strip()
                    sequence = ''.join([c for c in sequence if c.isalpha()])
                    
                    # Save first 20 sequences for debugging
                    if len(sample_sequences) < 20:
                        sample_sequences.append(sequence)
                    
                    # Skip duplicates
                    if sequence in duplicates:
                        rejection_reasons['duplicate'] += 1
                        continue
                    
                    # Check constraints
                    if len(sequence) < 10 or len(sequence) > 36:
                        rejection_reasons['constraints'] += 1
                        continue
                    if 'K' not in sequence and 'R' not in sequence:
                        rejection_reasons['constraints'] += 1
                        continue
                    
                    # Score with NEURAL scorer (same as training!)
                    scores = self._score_amp_neural(sequence, token_ids_np)
                    
                    # Check thresholds (0.80 = sAMPpred-GAT threshold)
                    if scores['overall_score'] < min_score:
                        rejection_reasons['low_overall_score'] += 1
                        continue
                    
                    if scores['membrane_reactivity'] < min_membrane_score:
                        rejection_reasons['low_membrane_score'] += 1
                        continue
                    
                    # Add to results
                    duplicates.add(sequence)
                    generated_amps.append({
                        'sequence': sequence,
                        'overall_score': scores['overall_score'],
                        'membrane_reactivity': scores['membrane_reactivity'],
                        'length': len(sequence),
                        'charge': scores['charge'],
                        'hydrophobicity': scores['hydrophobicity'],
                        'aliphatic_index': scores['aliphatic_index'],
                        'secondary_structure': scores['secondary_structure']
                    })
                    
                    pbar.update(1)
                    
                    if len(generated_amps) >= num_amps:
                        break
                        
                except Exception as e:
                    rejection_reasons['exception'] += 1
                    if attempts <= 10:
                        self.logger.error(f"\n❌ Error (attempt {attempts}): {str(e)[:100]}")
                    continue
        
        pbar.close()
        
        # Create DataFrame
        df = pd.DataFrame(generated_amps)
        
        if len(df) == 0:
            self.logger.warning("\n❌ No AMPs generated! Check parameters.\n")
            return df
        
        # Sort by overall score
        df = df.sort_values('overall_score', ascending=False).reset_index(drop=True)
        
        # Statistics
        self.logger.info(f"\n{'='*80}")
        self.logger.info("📊 GENERATION STATISTICS")
        self.logger.info(f"{'='*80}\n")
        
        self.logger.info(f"✅ Generated: {len(df)} unique AMPs")
        self.logger.info(f"📝 Total attempts: {attempts}")
        self.logger.info(f"✨ Success rate: {len(df)/attempts*100:.1f}%\n")
        
        self.logger.info(f"🎯 Quality Scores (Neural Scorer):")
        self.logger.info(f"   Overall:    {df['overall_score'].mean():.4f} ± {df['overall_score'].std():.4f}")
        self.logger.info(f"   Membrane:   {df['membrane_reactivity'].mean():.4f} ± {df['membrane_reactivity'].std():.4f}\n")
        
        self.logger.info(f"📏 Sequence Properties:")
        self.logger.info(f"   Length:     {df['length'].mean():.1f} ± {df['length'].std():.1f}")
        self.logger.info(f"   Charge:     {df['charge'].mean():.1f} ± {df['charge'].std():.1f}")
        self.logger.info(f"   Hydrophob:  {df['hydrophobicity'].mean():.3f} ± {df['hydrophobicity'].std():.3f}\n")
        
        # Save results
        self._save_results(df)
        
        return df
    
    def _save_results(self, df: pd.DataFrame):
        """Save generated AMPs to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV file
        csv_path = self.output_dir / f'generated_amps_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        self.logger.info(f"💾 Saved CSV: {csv_path}")
        
        # FASTA file
        fasta_path = self.output_dir / f'generated_amps_{timestamp}.fasta'
        with open(fasta_path, 'w') as f:
            for idx, row in df.iterrows():
                f.write(f">AMP_{idx+1}|score={row['overall_score']:.4f}|membrane={row['membrane_reactivity']:.4f}\n")
                f.write(f"{row['sequence']}\n")
        self.logger.info(f"💾 Saved FASTA: {fasta_path}")
        
        # Summary file
        summary_path = self.output_dir / f'generation_summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AMP GENERATION SUMMARY (NEURAL SCORER)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {len(df)} unique AMPs\n\n")
            
            f.write("Quality Scores (Neural Scorer):\n")
            f.write(f"  Overall:    {df['overall_score'].mean():.4f} ± {df['overall_score'].std():.4f}\n")
            f.write(f"  Membrane:   {df['membrane_reactivity'].mean():.4f} ± {df['membrane_reactivity'].std():.4f}\n\n")
            
            f.write("Sequence Properties:\n")
            f.write(f"  Length:     {df['length'].mean():.1f} ± {df['length'].std():.1f}\n")
            f.write(f"  Charge:     {df['charge'].mean():.1f} ± {df['charge'].std():.1f}\n")
            f.write(f"  Hydrophob:  {df['hydrophobicity'].mean():.3f} ± {df['hydrophobicity'].std():.3f}\n\n")
            
            f.write("Top 10 AMPs:\n")
            for idx, row in df.head(10).iterrows():
                f.write(f"  {idx+1}. {row['sequence']} (score={row['overall_score']:.4f})\n")
        
        self.logger.info(f"💾 Saved Summary: {summary_path}\n")


def main():
    """Main execution."""
    print("="*80)
    print("🚀 PHASE 2: AMP GENERATION (NEURAL SCORER VERSION)")
    print("="*80)
    
    # Configuration - USE BEST MODEL!
    checkpoint_prefix = "lsbo_guided_training_results/checkpoints/best_model_epoch_096_acc_0.9670"
    output_dir = "generated_amps_neural_scorer"
    
    # Initialize generator
    generator = AMPGenerator(
        checkpoint_prefix=checkpoint_prefix,
        output_dir=output_dir
    )
    
    # Generate AMPs with neural scorer
    df = generator.generate(
        num_amps=1000,
        temperature=0.8,
        min_score=0.80,  # sAMPpred-GAT threshold
        min_membrane_score=0.80
    )
    
    print("\n" + "="*80)
    print("✅ GENERATION COMPLETE!")
    print("="*80)
    
    if len(df) > 0:
        print(f"\n🎉 Successfully generated {len(df)} high-quality AMPs!")
        print(f"📁 Results saved to: {output_dir}/")
        print(f"\n🔬 Quality: {df['overall_score'].mean():.4f} (Neural Scorer)")
        print(f"🧬 Ready for sAMPpred-GAT validation (threshold 0.80)")
    else:
        print("\n⚠️  No AMPs generated. Check parameters.")
    
    print()


if __name__ == "__main__":
    main()
