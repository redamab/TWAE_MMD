# TWAE-MMD: Transformer-Wasserstein Autoencoder for Peptide Generation

This repository contains the implementation of a **Transformer-Wasserstein Autoencoder with Maximum Mean Discrepancy (TWAE-MMD)**, a deep learning 
framework developed for the *de novo* design of functional antimicrobial peptides (AMPs).

## Core Methodology
The framework utilizes a Transformer-based architecture regularized by **Wasserstein distance variants** and **Maximum Mean Discrepancy (MMD)**. 
This combination ensures a highly structured latent space, enabling precise control over the generation process and avoiding common pitfalls like posterior collapse.

## Implementation & Usage

The project provides two distinct strategies for peptide discovery and generation:

### 1. EMA: Baseline Random Sampling
The **Exponential Moving Average (EMA)** strategy serves as a stochastic baseline for sampling from the learned latent distribution.

*   **Execution:**
    ```bash
    cd EMA/docker/scripts/
    python twae_mmd_trainer_EMA.py
    ```

---

### 2. LSBO: Active Sampling Strategy
The **Latent Space Bayesian Optimization (LSBO)** strategy is the advanced core of this project, designed for targeted exploration of high-quality regions within the latent space.

#### 2.1 Training Phase
Train the model for 100 epochs to align the latent space distribution and identify high-quality functional regions.
*   **Execution:**
    ```bash
    python LSBO/docker/scripts/train_twae_mmd_LSBO.py
    ```
*   **Outcome:** At the end of this phase, the model is saved with optimal weights and a high-matching latent space distribution.

#### 2.2 Generation Phase
Utilize the trained model and a **Neural Scorer** to generate novel peptides from the optimized latent regions.
*   **Execution:**
    ```bash
    python LSBO/docker/scripts/generate_amps_NEURAL_SCORER.py
    ```

## Key Performance Metrics
*   **Validation Success Rate:** 99.4% for generated functional peptides.
*   **Efficiency:** LSBO demonstrates a **10-20x improvement** in discovery efficiency compared to random sampling baselines.
