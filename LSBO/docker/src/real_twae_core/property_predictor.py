"""
Property Predictor for AMP Quality Scoring
===========================================

Provides property predictors for scoring AMP quality during LSBO-guided training.

Includes:
1. ImprovedAMPScorer - Weighted heuristics based on AMP literature
2. SimplePropertyPredictor - Basic heuristics (legacy)
3. ModelBasedPropertyPredictor - Uses trained classifier (legacy)

"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional


class ImprovedAMPScorer:
    """
    Improved AMP scoring based on weighted heuristics from literature.
    
    """
    
    def __init__(self):
        # Amino acid properties
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        
        # Hydrophobic amino acids (Kyte-Doolittle scale)
        self.hydrophobic = set('AILMFVPWG')
        
        # Charged amino acids
        self.positive = set('KRH')
        self.negative = set('DE')
        
        # Polar uncharged
        self.polar = set('STNQ')
        
        # Aromatic
        self.aromatic = set('FWY')
        
        # Aliphatic (for thermal stability)
        self.aliphatic = set('AILV')
        
        # Alpha-helix formers (Chou-Fasman parameters)
        self.helix_formers = set('AELMK')
        
        # Beta-sheet formers
        self.sheet_formers = set('VIFY')
        
        # Hydrophobicity values (Kyte-Doolittle scale, normalized 0-1)
        self.hydrophobicity_values = {
            'A': 0.70, 'C': 0.78, 'D': 0.00, 'E': 0.00, 'F': 0.88,
            'G': 0.48, 'H': 0.08, 'I': 1.00, 'K': 0.00, 'L': 0.93,
            'M': 0.78, 'N': 0.09, 'P': 0.52, 'Q': 0.09, 'R': 0.00,
            'S': 0.42, 'T': 0.45, 'V': 0.86, 'W': 0.81, 'Y': 0.53
        }
        
        # Boman index values (protein-protein interaction potential)
        self.boman_values = {
            'A': 2.0, 'C': 1.0, 'D': -3.0, 'E': -3.0, 'F': 2.0,
            'G': 0.0, 'H': -1.0, 'I': 3.0, 'K': -2.0, 'L': 2.0,
            'M': 2.0, 'N': -1.0, 'P': 0.0, 'Q': -1.0, 'R': -2.0,
            'S': 0.0, 'T': 0.0, 'V': 2.0, 'W': 1.0, 'Y': 1.0
        }
    
    def _decode_sequence(self, sequence: tf.Tensor) -> str:
        """Convert token sequence to amino acid string."""
        seq_np = sequence.numpy()
        aa_indices = [int(idx) for idx in seq_np if idx > 4 and idx < 25]
        aa_string = ''.join([self.aa_list[idx - 5] for idx in aa_indices])
        return aa_string
    
    def _calculate_length_score(self, length: int) -> float:
        """Score based on optimal AMP length."""
        if 15 <= length <= 25:
            return 1.0
        elif 12 <= length < 15 or 25 < length <= 28:
            return 0.95
        elif 10 <= length < 12 or 28 < length <= 30:
            return 0.85
        elif 8 <= length < 10 or 30 < length <= 35:
            return 0.70
        else:
            return 0.50
    
    def _calculate_charge_score(self, net_charge: int) -> float:
        """Score based on net positive charge."""
        if 3 <= net_charge <= 5:
            return 1.0
        elif net_charge == 2 or net_charge == 6:
            return 0.90
        elif net_charge == 7:
            return 0.85
        elif net_charge == 1 or net_charge == 8:
            return 0.75
        elif net_charge == 9:
            return 0.70
        else:
            return 0.50
    
    def _calculate_hydrophobicity_score(self, hydro_ratio: float) -> float:
        """Score based on hydrophobic content."""
        if 0.40 <= hydro_ratio <= 0.50:
            return 1.0
        elif 0.35 <= hydro_ratio < 0.40 or 0.50 < hydro_ratio <= 0.55:
            return 0.95
        elif 0.30 <= hydro_ratio < 0.35 or 0.55 < hydro_ratio <= 0.60:
            return 0.85
        elif 0.25 <= hydro_ratio < 0.30 or 0.60 < hydro_ratio <= 0.65:
            return 0.75
        else:
            return 0.60
    
    def _calculate_amphipathicity_score(self, seq: str) -> float:
        """Score based on amphipathic character."""
        if len(seq) < 5:
            return 0.5
        
        # Calculate local hydrophobicity variance (window size 5)
        window_size = 5
        hydro_values = [self.hydrophobicity_values.get(aa, 0.5) for aa in seq]
        
        if len(hydro_values) < window_size:
            return 0.7
        
        windows = []
        for i in range(len(hydro_values) - window_size + 1):
            window = hydro_values[i:i+window_size]
            windows.append(np.mean(window))
        
        # High variance = good amphipathicity
        variance = np.var(windows)
        
        if variance > 0.08:
            return 1.0
        elif variance > 0.05:
            return 0.90
        elif variance > 0.03:
            return 0.80
        else:
            return 0.70
    
    def _calculate_boman_index(self, seq: str) -> float:
        """Calculate Boman index (protein-protein interaction potential)."""
        if len(seq) == 0:
            return 0.5
        
        boman_sum = sum(self.boman_values.get(aa, 0.0) for aa in seq)
        boman_index = boman_sum / len(seq)
        
        # Score based on Boman index
        if 1.5 <= boman_index <= 3.0:
            return 1.0
        elif 1.0 <= boman_index < 1.5 or 3.0 < boman_index <= 3.5:
            return 0.90
        elif 0.5 <= boman_index < 1.0 or 3.5 < boman_index <= 4.0:
            return 0.80
        else:
            return 0.70
    
    def _calculate_aliphatic_index(self, seq: str) -> float:
        """Calculate aliphatic index (thermal stability)."""
        if len(seq) == 0:
            return 0.5
        
        aliphatic_count = sum(1 for aa in seq if aa in self.aliphatic)
        aliphatic_ratio = aliphatic_count / len(seq)
        
        # Convert to index (0-100 scale)
        aliphatic_index = aliphatic_ratio * 100
        
        if 60 <= aliphatic_index <= 100:
            return 1.0
        elif 50 <= aliphatic_index < 60:
            return 0.90
        elif 40 <= aliphatic_index < 50:
            return 0.80
        else:
            return 0.70
    
    def _calculate_secondary_structure_score(self, seq: str) -> float:
        """Score based on secondary structure propensity."""
        if len(seq) == 0:
            return 0.5
        
        helix_count = sum(1 for aa in seq if aa in self.helix_formers)
        sheet_count = sum(1 for aa in seq if aa in self.sheet_formers)
        
        helix_ratio = helix_count / len(seq)
        sheet_ratio = sheet_count / len(seq)
        
        # Prefer alpha-helix (more common in AMPs)
        if helix_ratio > 0.5:
            return 1.0
        elif helix_ratio > 0.4 or sheet_ratio > 0.4:
            return 0.90
        elif helix_ratio > 0.3 or sheet_ratio > 0.3:
            return 0.80
        else:
            return 0.70
    
    def _calculate_aromatic_score(self, seq: str) -> float:
        """Score based on aromatic content."""
        if len(seq) == 0:
            return 0.5
        
        aromatic_count = sum(1 for aa in seq if aa in self.aromatic)
        aromatic_ratio = aromatic_count / len(seq)
        
        if 0.10 <= aromatic_ratio <= 0.25:
            return 1.0
        elif 0.05 <= aromatic_ratio < 0.10 or 0.25 < aromatic_ratio <= 0.30:
            return 0.85
        else:
            return 0.70
    
    def __call__(self, sequence: tf.Tensor) -> float:
        """
        Calculate comprehensive AMP score.
        
        Weighted components:
        - Length (15%): Optimal size for membrane interaction
        - Charge (20%): Electrostatic attraction to bacterial membranes
        - Hydrophobicity (15%): Membrane insertion capability
        - Amphipathicity (15%): Membrane disruption potential
        - Boman index (15%): Protein-protein interaction
        - Secondary structure (10%): Structural stability
        - Aliphatic index (5%): Thermal stability
        - Aromatic content (5%): Membrane anchoring
        
        Returns:
            Score between 0.0 and 1.0 (can reach 0.85-0.95 for ideal AMPs)
        """
        # Decode sequence
        seq = self._decode_sequence(sequence)
        
        if len(seq) == 0:
            return 0.1
        
        # Calculate individual scores
        length = len(seq)
        length_score = self._calculate_length_score(length)
        
        # Charge
        positive_count = sum(1 for aa in seq if aa in self.positive)
        negative_count = sum(1 for aa in seq if aa in self.negative)
        net_charge = positive_count - negative_count
        charge_score = self._calculate_charge_score(net_charge)
        
        # Hydrophobicity
        hydrophobic_count = sum(1 for aa in seq if aa in self.hydrophobic)
        hydro_ratio = hydrophobic_count / length
        hydro_score = self._calculate_hydrophobicity_score(hydro_ratio)
        
        # Amphipathicity
        amphipathicity_score = self._calculate_amphipathicity_score(seq)
        
        # Boman index
        boman_score = self._calculate_boman_index(seq)
        
        # Secondary structure
        structure_score = self._calculate_secondary_structure_score(seq)
        
        # Aliphatic index
        aliphatic_score = self._calculate_aliphatic_index(seq)
        
        # Aromatic content
        aromatic_score = self._calculate_aromatic_score(seq)
        
        # Weighted combination
        final_score = (
            0.15 * length_score +
            0.20 * charge_score +
            0.15 * hydro_score +
            0.15 * amphipathicity_score +
            0.15 * boman_score +
            0.10 * structure_score +
            0.05 * aliphatic_score +
            0.05 * aromatic_score
        )
        
        return final_score


class SimplePropertyPredictor:
    """
    Simple heuristic-based property predictor (legacy).
    
    For backward compatibility with existing code.
    """
    
    def __init__(self):
        self.aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.hydrophobic = set('AILMFVPWG')
        self.positive = set('KRH')
        self.negative = set('DE')
    
    def _decode_sequence(self, sequence: tf.Tensor) -> str:
        """Convert token sequence to amino acid string."""
        seq_np = sequence.numpy()
        aa_indices = [int(idx) for idx in seq_np if idx > 4 and idx < 25]
        aa_string = ''.join([self.aa_list[idx - 5] for idx in aa_indices])
        return aa_string
    
    def __call__(self, sequence: tf.Tensor) -> float:
        """Calculate simple heuristic score."""
        seq = self._decode_sequence(sequence)
        
        if len(seq) == 0:
            return 0.1
        
        length = len(seq)
        
        # Count properties
        hydrophobic_count = sum(1 for aa in seq if aa in self.hydrophobic)
        positive_count = sum(1 for aa in seq if aa in self.positive)
        negative_count = sum(1 for aa in seq if aa in self.negative)
        
        # Length score
        if 10 <= length <= 30:
            length_score = 1.0
        elif 8 <= length <= 35:
            length_score = 0.8
        else:
            length_score = 0.5
        
        # Charge score
        net_charge = positive_count - negative_count
        if 2 <= net_charge <= 6:
            charge_score = 1.0
        elif 1 <= net_charge <= 8:
            charge_score = 0.8
        else:
            charge_score = 0.5
        
        # Hydrophobicity score
        hydro_ratio = hydrophobic_count / length if length > 0 else 0
        if 0.3 <= hydro_ratio <= 0.6:
            hydro_score = 1.0
        elif 0.2 <= hydro_ratio <= 0.7:
            hydro_score = 0.8
        else:
            hydro_score = 0.6
        
        # Combined score
        score = 0.3 * length_score + 0.4 * charge_score + 0.3 * hydro_score
        
        return score


class ModelBasedPropertyPredictor:
    """
    Model-based property predictor using trained classifier (legacy).
    
    For backward compatibility with existing code.
    """
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, sequence: tf.Tensor) -> float:
        """Calculate score using trained classifier."""
        try:
            seq_batch = tf.expand_dims(sequence, 0)
            latent_vector, _, _ = self.model.encoder(seq_batch, training=False)
            logits = self.model.classifier(latent_vector, training=False)
            probabilities = tf.nn.softmax(logits, axis=-1)
            amp_probability = float(probabilities[0, 1].numpy())
            return amp_probability
        except Exception:
            return 0.5
