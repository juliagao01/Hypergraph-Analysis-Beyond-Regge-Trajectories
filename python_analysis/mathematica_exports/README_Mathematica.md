# Mathematica Integration Guide

This directory contains all exported data and plotting functions for Mathematica analysis.

## Files:

- `unified_states.m`: Main data model with all particle states and features
- `regge_fit_results.m`: Regge trajectory fit parameters and diagnostics
- `bridging_analysis_results.m`: Statistical analysis linking hypergraph features to Regge residuals
- `hypothesis_test_results.m`: Results from pre-registered hypothesis tests
- `predictions.m`: Predictions with hypergraph-informed confidence levels
- `unified_analysis_plots.nb`: Mathematica notebook with publication-quality plotting functions

## Usage:

1. Open `unified_analysis_plots.nb` in Mathematica
2. Run the notebook to generate publication-quality plots
3. All plots are automatically exported as high-resolution PDFs

## Key Functions:

- `getStates[]`: Access all particle data
- `reggeFit[x]`: Regge trajectory fit function
- `getCorrelation[feature]`: Get correlation coefficient for a feature
- `isHypothesisSupported[hypothesis]`: Check if a hypothesis is supported
- `getPredictions[]`: Access predictions with confidence levels
