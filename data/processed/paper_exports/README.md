# Regge Trajectory Analysis - Reproducibility Package

**Analysis Date:** 2025-08-16T20:48:06.295229
**PDG Snapshot Date:** 2024-01-01

## Contents

This package contains all results, figures, and metadata needed to reproduce the analysis.

### Data Files
- `regge_fit_parameters.json`: Fitted Regge trajectory parameters
- `mass_predictions.csv`: Mass predictions with uncertainties
- `pdg_cross_check_results.csv`: PDG cross-check results
- `regge_diagnostics.json`: Fit diagnostics and residuals
- `hypergraph_metrics.json`: Hypergraph analysis metrics
- `motif_analysis.json`: Motif and cycle analysis results

### Comparison Results
- `method_comparison.csv`: Hypergraph vs baseline comparison
- `performance_metrics.json`: Performance benchmarking results
- `baseline_analysis.json`: Traditional analysis results

### Stability Analysis
- `stability_metrics.json`: Stability under PDG updates
- `data_changes.json`: Changes between PDG versions
- `reclassification_results.json`: Reclassification scenarios

### Metadata
- `provenance_report.txt`: Complete provenance information
- `provenance.json`: Machine-readable provenance data
- `export_summary.csv`: Summary of all exported files

## Reproducibility

To reproduce this analysis:

1. Install required software versions (see provenance.json)
2. Use the same analysis settings (see provenance.json)
3. Run the analysis pipeline with the provided data
4. Compare results with the exported files

## Software Requirements

- python: 3.13.2 (main, Feb  4 2025, 14:51:09) [Clang 15.0.0 (clang-1500.1.0.2.5)]
- numpy: 2.3.2
- pandas: 2.3.1
- matplotlib: 3.10.5
- scipy: 1.16.1
- scikit-learn: 1.7.1
- networkx: 3.5
