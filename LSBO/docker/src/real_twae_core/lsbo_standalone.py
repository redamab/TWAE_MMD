"""
Standalone LSBO Implementation
==========================================================

This module provides LSBO without requiring scikit-optimize.
Uses only numpy and scipy which are already available.


"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from scipy.stats import norm
from scipy.optimize import minimize

try:
    from .constraints import AMPConstraints
except ImportError:
    AMPConstraints = None


class SimpleGaussianProcess:
    """
    Simple Gaussian Process implementation without scikit-learn.
    
    Uses a simple RBF kernel and direct matrix inversion.
    """
    
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        """
        Initialize Gaussian Process.
        
        Args:
            length_scale: RBF kernel length scale
            noise: Noise level for numerical stability
        """
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        # Compute pairwise squared distances
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances_sq = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        
        # RBF kernel
        K = np.exp(-distances_sq / (2 * self.length_scale**2))
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Gaussian Process to data.
        
        Args:
            X: Training inputs [n_samples, n_features]
            y: Training outputs [n_samples]
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).reshape(-1, 1)
        
        # Compute kernel matrix
        K = self._rbf_kernel(self.X_train, self.X_train)
        
        # Add noise for numerical stability
        K += self.noise * np.eye(len(X))
        
        # Invert kernel matrix
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudo-inverse
            self.K_inv = np.linalg.pinv(K)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at new points.
        
        Args:
            X: Test inputs [n_samples, n_features]
            
        Returns:
            mean: Predicted means [n_samples]
            std: Predicted standard deviations [n_samples]
        """
        if self.X_train is None:
            # No training data, return zeros
            return np.zeros(len(X)), np.ones(len(X))
        
        X = np.array(X)
        
        # Compute kernel between test and training points
        K_star = self._rbf_kernel(X, self.X_train)
        
        # Compute mean
        mean = np.dot(K_star, np.dot(self.K_inv, self.y_train)).flatten()
        
        # Compute variance
        K_star_star = self._rbf_kernel(X, X)
        var = np.diag(K_star_star) - np.sum(K_star * np.dot(K_star, self.K_inv), axis=1)
        var = np.maximum(var, 1e-10)  # Ensure positive
        std = np.sqrt(var)
        
        return mean, std


class StandaloneLSBO:
    """
    Standalone LSBO without external dependencies.
    
    Uses only numpy and scipy (already available).
    """
    
    def __init__(self,
                 model,
                 property_predictor: Optional[Callable] = None,
                 constraints: Optional['AMPConstraints'] = None,
                 acquisition: str = 'ei',
                 xi: float = 0.01,
                 kappa: float = 2.576):
        """
        Initialize standalone LSBO.
        
        Args:
            model: Trained TWAE model
            property_predictor: Function to predict properties
            constraints: Biological constraints
            acquisition: Acquisition function ('ei', 'ucb', 'poi')
            xi: Exploration parameter for EI
            kappa: Exploration parameter for UCB
        """
        self.model = model
        self.latent_dim = model.latent_manager.latent_dim
        self.property_predictor = property_predictor
        self.constraints = constraints
        self.xi = xi
        self.kappa = kappa
        self.acquisition_type = acquisition
        
        # Initialize simple Gaussian Process
        self.gp = SimpleGaussianProcess(length_scale=1.0, noise=1e-6)
        
        # History
        self.X_history = []
        self.y_history = []
    
    def optimize(self,
                 num_iterations: int = 100,
                 num_initial_samples: int = 20,
                 top_k: int = 10,
                 verbose: bool = False) -> Tuple[List[np.ndarray], tf.Tensor]:
        """
        Run LSBO optimization.
        
        Args:
            num_iterations: Number of optimization iterations
            num_initial_samples: Number of initial random samples
            top_k: Number of top sequences to return
            verbose: Print progress
            
        Returns:
            best_latents: Top k latent vectors
            best_sequences: Top k decoded sequences
        """
        if verbose:
            print(f"\n Starting Standalone LSBO Optimization")
            print(f"   Iterations: {num_iterations}")
            print(f"   Initial samples: {num_initial_samples}")
            print(f"   Acquisition: {self.acquisition_type}")
            if self.constraints:
                print(f"   Constraints: Enabled")
        
        # Phase 1: Initial random sampling
        if verbose:
            print(f"\n Phase 1: Initial Random Sampling ({num_initial_samples} samples)")
        
        X_init = self._sample_initial(num_initial_samples)
        y_init = self._evaluate_batch(X_init, verbose=verbose)
        
        self.X_history = X_init
        self.y_history = y_init
        
        # Phase 2: Bayesian optimization with random exploration
        if verbose:
            print(f"\n Phase 2: Bayesian Optimization ({num_iterations - num_initial_samples} iterations)")
            print(f"   Strategy: 90% BO + 10% Random Exploration")
        
        for i in range(num_iterations - num_initial_samples):
            # Add 10% random exploration to prevent getting stuck
            if i % 10 == 0 and i > 0:
                # Random exploration
                x_next = self._sample_initial(1)[0]
                if verbose:
                    print(f"  Random exploration at iteration {i+1}")
            else:
                # Bayesian optimization
                # Fit GP
                self.gp.fit(self.X_history, self.y_history)
                
                # Find next point
                x_next = self._propose_location()
            
            # Evaluate
            y_next = self._evaluate_batch([x_next], verbose=False)[0]
            
            # Update history
            self.X_history.append(x_next)
            self.y_history.append(y_next)
            
            if verbose and (i + 1) % 10 == 0:
                best_y = max(self.y_history)
                print(f"  Iteration {i+1}/{num_iterations - num_initial_samples}: Best score = {best_y:.4f}")
        
        # Get top k results
        if verbose:
            print(f"\n Optimization Complete!")
            print(f"   Total evaluations: {len(self.y_history)}")
            print(f"   Best score: {max(self.y_history):.4f}")
        
        # Sort by score
        sorted_indices = np.argsort(self.y_history)[::-1]
        top_indices = sorted_indices[:top_k]
        
        best_latents = [self.X_history[i] for i in top_indices]
        
        # Decode sequences
        decoded_sequences = [self._decode_single(z) for z in best_latents]
        
        # Pad sequences to same length before stacking
        max_len = max(seq.shape[0] for seq in decoded_sequences)
        padded_sequences = []
        for seq in decoded_sequences:
            if seq.shape[0] < max_len:
                # Pad with zeros (PAD token)
                padding = tf.zeros([max_len - seq.shape[0]], dtype=seq.dtype)
                padded_seq = tf.concat([seq, padding], axis=0)
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        best_sequences = tf.stack(padded_sequences)
        
        return best_latents, best_sequences
    
    def _sample_initial(self, num_samples: int) -> List[np.ndarray]:
        """Sample initial latent vectors from learned distribution."""
        # Use EMA statistics from trained model
        latent_mean = self.model.latent_manager.latent_mean.numpy()
        latent_std = self.model.latent_manager.latent_std.numpy()
        
        # Sample from Gaussian
        latent_samples = tf.random.normal(
            [num_samples, self.latent_dim],
            mean=latent_mean,
            stddev=latent_std,
            dtype=tf.float32
        )
        return [z.numpy() for z in latent_samples]
    
    def _evaluate_batch(self, X: List[np.ndarray], verbose: bool = False) -> List[float]:
        """Evaluate a batch of latent vectors."""
        scores = []
        for i, z in enumerate(X):
            # Decode to sequence
            sequence = self._decode_single(z)
            
            # Check constraints first
            if self.constraints is not None:
                if not self.constraints.satisfies(sequence):
                    # Penalty for invalid sequences
                    scores.append(-1000.0)
                    continue
            
            # Predict properties
            if self.property_predictor is not None:
                score = self.property_predictor(sequence)
            else:
                # Default: use reconstruction quality
                score = self._default_score(z, sequence)
            
            scores.append(score)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{len(X)} samples")
        
        return scores
    
    def _decode_single(self, z: np.ndarray) -> tf.Tensor:
        """Decode single latent vector to sequence."""
        z_tensor = tf.constant(z.reshape(1, -1), dtype=tf.float32)
        # Use decoder's generate method for autoregressive generation
        sequence = self.model.decoder.generate(
            start_token_id=1,  # CLS token
            latent_vector=z_tensor,
            max_length=self.model.config.max_length,
            temperature=1.0
        )
        return sequence[0]
    
    def _default_score(self, z: np.ndarray, sequence: tf.Tensor) -> float:
        """Default scoring based on reconstruction quality."""
        # Encode back
        z_reconstructed = self.model.encoder.encode(
            tf.expand_dims(sequence, 0),
            training=False
        )[0].numpy()
        
        # Score = negative reconstruction error
        reconstruction_error = np.mean((z - z_reconstructed)**2)
        return -reconstruction_error
    
    def _propose_location(self) -> np.ndarray:
        """Propose next location using acquisition function."""
        # Get current best
        y_max = max(self.y_history)
        
        # Define acquisition function
        def acquisition(x):
            x = x.reshape(1, -1)
            mean, std = self.gp.predict(x)
            
            if self.acquisition_type == 'ei':
                # Expected Improvement
                z = (mean[0] - y_max - self.xi) / (std[0] + 1e-9)
                ei = (mean[0] - y_max - self.xi) * norm.cdf(z) + std[0] * norm.pdf(z)
                return -ei  # Minimize negative
            
            elif self.acquisition_type == 'ucb':
                # Upper Confidence Bound
                ucb = mean[0] + self.kappa * std[0]
                return -ucb  # Minimize negative
            
            elif self.acquisition_type == 'poi':
                # Probability of Improvement
                z = (mean[0] - y_max - self.xi) / (std[0] + 1e-9)
                poi = norm.cdf(z)
                return -poi  # Minimize negative
            
            else:
                raise ValueError(f"Unknown acquisition: {self.acquisition_type}")
        
        # Optimize acquisition function
        # Use multiple random starts
        best_x = None
        best_acq = float('inf')
        
        for _ in range(10):
            # Random start
            x0 = np.random.randn(self.latent_dim)
            
            # Optimize
            result = minimize(
                acquisition,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return best_x


class MultiObjectiveStandaloneLSBO(StandaloneLSBO):
    """
    Multi-objective standalone LSBO.
    """
    
    def __init__(self,
                 model,
                 property_predictors: Dict[str, Callable],
                 property_weights: Dict[str, float],
                 constraints: Optional['AMPConstraints'] = None,
                 **kwargs):
        """
        Initialize multi-objective standalone LSBO.
        
        Args:
            model: Trained TWAE model
            property_predictors: Dict of property prediction functions
            property_weights: Dict of property weights
            constraints: Biological constraints
        """
        super().__init__(model, None, constraints, **kwargs)
        
        self.property_predictors = property_predictors
        self.property_weights = property_weights
        
        # Override property predictor with weighted combination
        self.property_predictor = self._weighted_predictor
    
    def _weighted_predictor(self, sequence: tf.Tensor) -> float:
        """Weighted combination of multiple properties."""
        total_score = 0.0
        
        for prop_name, predictor in self.property_predictors.items():
            weight = self.property_weights.get(prop_name, 0.0)
            score = predictor(sequence)
            total_score += weight * score
        
        return total_score



