# Hypergraph-Based Regge Trajectory Analysis

A novel framework for analyzing hadron spectroscopy by bridging Regge trajectory fitting with hypergraph-based network analysis of particle decay patterns.

## Overview

This repository contains the complete analysis code, data, and documentation for the paper "Hypergraph-Based Decay Analysis for Predicting and Classifying States Beyond Regge Trajectories".

Traditional Regge trajectory analysis provides powerful insights into hadron spectra but offers no systematic way to assess prediction reliability. This work introduces a **confidence-graded prediction framework** that combines:

- **Rigorous trajectory fitting**: Orthogonal distance regression (ODR) with bootstrap uncertainty quantification
- **Network topology analysis**: Hypergraph representations of decay channels with structural feature extraction
- **Reliability assessment**: Confidence scores based on decay-network coherence for predicted missing states

## Features

### Trajectory Analysis
- Weighted least squares (WLS) and orthogonal distance regression (ODR)
- Bootstrap resampling with systematic uncertainty propagation
- Residual diagnostics and influence analysis
- Leave-one-out cross-validation

### Hypergraph Construction
- Multi-body decay representation as hypergraphs
- Bipartite incidence graphs and co-occurrence projections
- Community detection via Louvain clustering
- Motif analysis with randomized baseline comparisons

### Network Metrics
- **Community purity**: Compositional coherence of decay clusters
- **Motif enrichment**: Z-scores for recurring topological patterns
- **Product entropy**: Shannon entropy of decay channel diversity
- Standard network measures: degree, clustering, assortativity

### Prediction Framework
- Confidence-graded missing resonance predictions
- Structural reliability scores based on decay topology
- Prioritized experimental search targets

## Requirements

### Software
- **Wolfram Mathematica 13.0+** (primary analysis environment)
- Particle Data Group (PDG) data access via `ParticleData` framework

### Mathematica Packages
All required functions are built-in to Mathematica 13.0+:
- `GraphUtilities`
- `CommunityGraphPlot`
- `FindGraphCommunities`
- `StatisticsLibrary` (for ODR and bootstrap)

## Installation & Usage

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/regge-hypergraph-analysis.git
cd regge-hypergraph-analysis
```

2. **Open Mathematica and load the analysis pipeline**
```mathematica
(* Set working directory *)
SetDirectory[NotebookDirectory[]];

(* Load core functions *)
Get["src/regge_fitting.wl"]
Get["src/hypergraph_tools.wl"]
Get["src/metrics.wl"]

(* Run complete analysis *)
Get["notebooks/01_data_preparation.nb"]
Get["notebooks/02_regge_fitting.nb"]
Get["notebooks/03_hypergraph_construction.nb"]
Get["notebooks/04_bridging_analysis.nb"]
Get["notebooks/05_predictions.nb"]
```

### Reproducing the Paper

To exactly reproduce all results in the paper:

1. **Use the frozen PDG snapshot** (July 12, 2025):
```mathematica
pdgData = Import["data/pdg_snapshot_2025-07-12.csv"];
```

2. **Run the complete pipeline** in sequence (notebooks 01-05)

3. **Figures and tables** will be generated in `figures/` and `results/`

All random processes (bootstrap resampling, community detection) use fixed seeds for reproducibility.

## Key Concepts

### Regge Trajectories
Linear relations between hadron spin J and squared mass M²:
```
J = α₀ + α' M²
```
where α' ≈ 0.9 GeV⁻² for mesons, slightly lower for baryons.

### Hypergraph Representation
Decay channels are represented as hyperedges connecting one resonance to multiple products:
```
Δ(1232) → {p, π⁰}
Δ(1600) → {p, π⁰}, {n, π⁺}, {Δ, π, π}
```

### Community Purity
Fraction of decay products in each cluster sharing the same particle type:
```
P = (1/Nₒ) Σᵢ (n_max,i / nᵢ)
```

### Confidence Scoring
Predicted states receive reliability scores based on:
- Community purity of associated decay cluster
- Extrapolation distance from fitted data
- Network coherence metrics

## License
This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Code: MIT License  
Data: CC BY 4.0  
Figures: CC BY 4.0

## Acknowledgments

- **Joseph Brennan** (Wolfram Research Europe Ltd) - Conceptualization and technical consultation
- **Stephen Wolfram** - Initial idea
- **Particle Data Group** - Hadron data (https://pdg.lbl.gov/)

## Contact

**Julia Gao**  
Fairview High School, Colorado State University  
Email: gaojulia01@gmail.com

---

*This repository contains supplementary materials for the paper submitted to PLOS One. For questions about the methodology or to request additional data, please contact the author.*
